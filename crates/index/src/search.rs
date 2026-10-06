//! Поиск по построенному индексу: перебор косинусом по всем векторам
//! стратегии.
//!
//! Векторы в базе L2-нормализованы (`embed::normalize`), поэтому косинус —
//! скалярное произведение. На сотнях чанков перебор в памяти занимает
//! миллисекунды (см. README, «SQLite, а не FAISS»).
//!
//! Модели не смешиваются: вопрос эмбеддится моделью, с которой запущен
//! сервер, и если она не совпала с моделью векторов стратегии — ошибка до
//! обращения к Ollama. Вектор другой модели имел бы другую размерность или
//! просто другое пространство, а косинус между ними ничего не значил бы.

use crate::embed::{EmbedConfig, Embedder};
use crate::ollama;
use crate::store::Store;
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::path::Path;

pub const DEFAULT_TOP_K: usize = 5;
pub const DEFAULT_CANDIDATE_TOP_K: usize = 20;
pub const DEFAULT_COMPARE_THRESHOLD: f32 = 0.5;
pub const MAX_TOP_K: usize = 20;
pub const MAX_CANDIDATE_TOP_K: usize = 100;

/// Чем эмбеддится вопрос.
#[derive(Debug, Clone)]
pub struct QueryModel {
    pub url: String,
    pub model: String,
    pub prefix: String,
    pub num_ctx: u32,
    pub batch: usize,
    pub rewrite_model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    pub chunk_id: String,
    pub source: String,
    pub section: String,
    pub score: f32,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct SearchOutput {
    pub query: String,
    pub used_query: String,
    pub strategy: String,
    pub model: String,
    pub dim: usize,
    pub top_k: usize,
    pub candidate_top_k: usize,
    pub similarity_threshold: Option<f32>,
    pub rewrite: bool,
    pub rewrite_fallback: Option<String>,
    pub candidates: usize,
    pub results: usize,
    pub hits: Vec<Hit>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOptions {
    pub top_k: usize,
    pub candidate_top_k: usize,
    pub similarity_threshold: Option<f32>,
}

pub fn validate_options(options: SearchOptions) -> Result<()> {
    ensure!(
        (1..=MAX_TOP_K).contains(&options.top_k),
        "top_k должен быть от 1 до {MAX_TOP_K}"
    );
    ensure!(
        (1..=MAX_CANDIDATE_TOP_K).contains(&options.candidate_top_k),
        "candidate_top_k должен быть от 1 до {MAX_CANDIDATE_TOP_K}"
    );
    if let Some(threshold) = options.similarity_threshold {
        ensure!(
            (-1.0..=1.0).contains(&threshold),
            "similarity_threshold должен быть от -1 до 1"
        );
    }
    Ok(())
}

fn retrieve(
    chunks: Vec<crate::store::StoredChunk>,
    vector: &[f32],
    options: SearchOptions,
) -> (usize, Vec<(f32, crate::store::ChunkRow)>) {
    let mut scored: Vec<(f32, usize, crate::store::ChunkRow)> = chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| (dot(vector, &chunk.vector), index, chunk.row))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.truncate(options.candidate_top_k);
    let candidates = scored.len();
    let mut hits = scored
        .into_iter()
        .filter(|(score, _, _)| options.similarity_threshold.is_none_or(|min| *score >= min))
        .map(|(score, _, row)| (score, row))
        .collect::<Vec<_>>();
    hits.truncate(options.top_k);
    (candidates, hits)
}

pub async fn search(
    db: &Path,
    model: &QueryModel,
    query: &str,
    strategy: Option<&str>,
    options: SearchOptions,
    rewrite: bool,
) -> Result<SearchOutput> {
    ensure!(!query.trim().is_empty(), "пустой запрос");
    validate_options(options)?;
    // Без проверки Store::open создал бы пустую базу и поиск «ничего не нашёл».
    ensure!(
        db.is_file(),
        "нет базы {}: сначала index_build (index-mcp build)",
        db.display()
    );
    let store = Store::open(db).await?;
    let builds = store.builds().await?;
    ensure!(
        !builds.is_empty(),
        "в {} нет построенных стратегий: сначала index_build",
        db.display()
    );
    let names = builds
        .iter()
        .map(|b| b.strategy.as_str())
        .collect::<Vec<_>>();
    let strategy = match strategy {
        Some(name) => name,
        None if names.len() == 1 => names[0],
        None => bail!(
            "в базе несколько стратегий ({}): укажи strategy",
            names.join(", ")
        ),
    };
    let build = builds
        .iter()
        .find(|b| b.strategy == strategy)
        .with_context(|| {
            format!(
                "стратегии {strategy} нет в базе; есть: {}",
                names.join(", ")
            )
        })?;
    ensure!(
        build.model == model.model,
        "стратегия {strategy} построена моделью {} (dim {}), а вопрос эмбеддится моделью {}: \
         векторы несравнимы — задай модель {} (index-mcp serve --model, в agentcli — index_model) или пересобери индекс",
        build.model,
        build.dim,
        model.model,
        build.model
    );
    let dim = build.dim as usize;
    let chunks = store.load(strategy).await?;
    // builds хранит последнее построение, а модель записана и при каждом
    // векторе: сверка по векторам ловит и базу, собранную частями.
    if let Some(bad) = chunks
        .iter()
        .find(|c| c.model != model.model || c.vector.len() != dim)
    {
        bail!(
            "вектор чанка {} построен моделью {} (dim {}), а не {} (dim {}): пересобери индекс",
            bad.row.chunk_id,
            bad.model,
            bad.vector.len(),
            model.model,
            dim
        );
    }
    let embedder = Embedder::new(EmbedConfig {
        url: model.url.clone(),
        model: model.model.clone(),
        batch: model.batch,
        num_ctx: model.num_ctx,
        dim,
    })?;
    let (used_query, rewrite_fallback) = if rewrite {
        match ollama::rewrite_query(
            &model.url,
            model.rewrite_model.as_deref().unwrap_or(&model.model),
            query,
        )
        .await
        {
            Ok(rewritten) => (rewritten, None),
            Err(err) => (query.to_string(), Some(format!("{err:#}"))),
        }
    } else {
        (query.to_string(), None)
    };
    let vector = embedder
        .embed(&model.prefix, &[&used_query])
        .await
        .context("эмбеддинг запроса")?
        .remove(0);
    let (candidates, scored) = retrieve(chunks, &vector, options);
    let results = scored.len();
    Ok(SearchOutput {
        query: query.to_string(),
        used_query,
        strategy: strategy.to_string(),
        model: model.model.clone(),
        dim,
        top_k: options.top_k,
        candidate_top_k: options.candidate_top_k,
        similarity_threshold: options.similarity_threshold,
        rewrite,
        rewrite_fallback,
        candidates,
        results,
        hits: scored
            .into_iter()
            .map(|(score, row)| Hit {
                chunk_id: row.chunk_id,
                source: row.source,
                section: row.section,
                score,
                text: row.text,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ChunkRow;

    fn chunk(id: &str, vector: Vec<f32>) -> crate::store::StoredChunk {
        crate::store::StoredChunk {
            row: ChunkRow {
                chunk_id: id.into(),
                strategy: "s".into(),
                source: "a.docx".into(),
                title: "A".into(),
                section: id.into(),
                ordinal: 0,
                char_start: 0,
                char_end: 1,
                text: id.into(),
                crosses_section: false,
                cut_mid_sentence: false,
            },
            model: "m".into(),
            vector,
        }
    }

    #[test]
    fn candidate_count_filters_low_scores_and_orders_ties_stably() {
        let chunks = vec![
            chunk("third", vec![0.9, 0.0]),
            chunk("first", vec![1.0, 0.0]),
            chunk("second", vec![0.9, 0.0]),
        ];
        let (candidates, hits) = retrieve(
            chunks,
            &[1.0, 0.0],
            SearchOptions {
                top_k: 2,
                candidate_top_k: 3,
                similarity_threshold: Some(0.85),
            },
        );
        assert_eq!(candidates, 3);
        assert_eq!(hits[0].1.chunk_id, "first");
        assert_eq!(hits[1].1.chunk_id, "third");
        assert_eq!(hits.len(), 2);

        let (_, empty) = retrieve(
            vec![chunk("low", vec![0.0, 1.0])],
            &[1.0, 0.0],
            SearchOptions {
                top_k: 5,
                candidate_top_k: 5,
                similarity_threshold: Some(0.5),
            },
        );
        assert!(empty.is_empty());
    }

    #[test]
    fn options_validate_bounds() {
        for options in [
            SearchOptions {
                top_k: 0,
                candidate_top_k: 1,
                similarity_threshold: None,
            },
            SearchOptions {
                top_k: 1,
                candidate_top_k: 0,
                similarity_threshold: None,
            },
            SearchOptions {
                top_k: 1,
                candidate_top_k: 1,
                similarity_threshold: Some(2.0),
            },
        ] {
            assert!(validate_options(options).is_err());
        }
    }
}
