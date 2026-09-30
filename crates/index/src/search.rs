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
use crate::store::Store;
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::path::Path;

/// Чем эмбеддится вопрос.
#[derive(Debug, Clone)]
pub struct QueryModel {
    pub url: String,
    pub model: String,
    pub prefix: String,
    pub num_ctx: u32,
    pub batch: usize,
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
    pub strategy: String,
    pub model: String,
    pub dim: usize,
    pub hits: Vec<Hit>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub async fn search(
    db: &Path,
    model: &QueryModel,
    query: &str,
    strategy: Option<&str>,
    top_k: usize,
) -> Result<SearchOutput> {
    ensure!(!query.trim().is_empty(), "пустой запрос");
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
         векторы несравнимы — запусти поиск с --model {} или пересобери индекс",
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
    let vector = embedder
        .embed(&model.prefix, &[query])
        .await
        .context("эмбеддинг запроса")?
        .remove(0);
    let mut scored: Vec<(f32, _)> = chunks
        .into_iter()
        .map(|c| (dot(&vector, &c.vector), c.row))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(top_k);
    Ok(SearchOutput {
        query: query.to_string(),
        strategy: strategy.to_string(),
        model: model.model.clone(),
        dim,
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
