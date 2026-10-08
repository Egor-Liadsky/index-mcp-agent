//! Сравнение стратегий: форма чанков и качество поиска по вопросам.
//!
//! Попадание — совпадение `section` найденного чанка с ожидаемым разделом
//! вопроса. Критерий строгий: если стратегия склеила маленький раздел с
//! соседним, его пути среди её чанков нет и вопрос к нему попасть не
//! может. Такие вопросы считаются отдельно («раздела нет среди чанков»),
//! чтобы эта часть разницы между стратегиями была видна, а не пряталась
//! в hit@k.

use crate::search::{SearchOptions, retrieve};
use crate::store::{BuildInfo, StoredChunk};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    pub question: String,
    /// Путь раздела `Раздел > Подраздел` или список равноценных путей.
    pub expected_section: Expected,
    /// Имя файла, если одинаковые разделы есть в разных конспектах.
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Expected {
    One(String),
    Any(Vec<String>),
}

impl Expected {
    fn matches(&self, section: &str) -> bool {
        match self {
            Expected::One(s) => s == section,
            Expected::Any(list) => list.iter().any(|s| s == section),
        }
    }

    fn display(&self) -> String {
        match self {
            Expected::One(s) => s.clone(),
            Expected::Any(list) => list.join(" | "),
        }
    }
}

impl Question {
    fn hits(&self, chunk: &StoredChunk) -> bool {
        let source_ok = self.source.as_deref().is_none_or(|want| {
            Path::new(&chunk.row.source)
                .file_name()
                .is_some_and(|name| name.to_string_lossy() == want)
        });
        source_ok && self.expected_section.matches(&chunk.row.section)
    }
}

pub fn load_questions(path: &Path) -> Result<Vec<Question>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("не прочитать {}", path.display()))?;
    let questions: Vec<Question> = serde_json::from_str(&text)
        .with_context(|| format!("{}: ожидается массив вопросов", path.display()))?;
    ensure!(
        !questions.is_empty(),
        "{}: нет ни одного вопроса",
        path.display()
    );
    Ok(questions)
}

/// Длины чанков в символах.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LenStats {
    pub min: usize,
    pub median: usize,
    pub p95: usize,
    pub max: usize,
}

/// Перцентили по ближайшему рангу: значение всегда — длина реального чанка.
pub fn length_stats(lengths: &[usize]) -> LenStats {
    if lengths.is_empty() {
        return LenStats::default();
    }
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let rank = |q: f64| {
        let idx = (q * sorted.len() as f64).ceil() as usize;
        sorted[idx.clamp(1, sorted.len()) - 1]
    };
    LenStats {
        min: sorted[0],
        median: rank(0.5),
        p95: rank(0.95),
        max: sorted[sorted.len() - 1],
    }
}

#[derive(Debug, Clone)]
pub struct Evaluation {
    pub strategy: String,
    pub build: Option<BuildInfo>,
    pub chunks: usize,
    pub lengths: LenStats,
    pub crosses: usize,
    pub cut: usize,
    /// Ранг первого попадания (с 1) для каждого вопроса или `None`.
    pub ranks: Vec<Option<usize>>,
    /// Вопросов, чей ожидаемый раздел не встречается ни в одном чанке.
    pub unreachable: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeParams {
    pub rewrite: bool,
    pub candidate_top_k: usize,
    pub top_k: usize,
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ModeEvaluation {
    pub strategy: String,
    pub mode: String,
    pub params: ModeParams,
    pub ranks: Vec<Option<usize>>,
    pub candidates_before: Vec<usize>,
    pub results_after: Vec<usize>,
    pub filtered_all: usize,
    pub rewrite_fallbacks: usize,
}

impl ModeEvaluation {
    pub fn hit_at(&self, k: usize) -> f64 {
        ratio(
            self.ranks
                .iter()
                .filter(|r| r.is_some_and(|r| r <= k))
                .count(),
            self.ranks.len(),
        )
    }

    pub fn mrr(&self) -> f64 {
        let sum: f64 = self
            .ranks
            .iter()
            .map(|rank| rank.map_or(0.0, |rank| 1.0 / rank as f64))
            .sum();
        if self.ranks.is_empty() {
            0.0
        } else {
            sum / self.ranks.len() as f64
        }
    }

    pub fn no_results(&self) -> usize {
        self.ranks.iter().filter(|rank| rank.is_none()).count()
    }

    pub fn average_candidates(&self) -> f64 {
        average(&self.candidates_before)
    }

    pub fn average_results(&self) -> f64 {
        average(&self.results_after)
    }
}

fn average(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

/// Оценивает одну комбинацию rewrite и threshold на одном наборе вопросов.
pub fn evaluate_mode(
    strategy: &str,
    chunks: &[StoredChunk],
    questions: &[Question],
    query_vectors: &[Vec<f32>],
    used_queries: &[String],
    params: ModeParams,
    rewrite_fallbacks: usize,
) -> ModeEvaluation {
    let mut ranks = Vec::with_capacity(questions.len());
    let mut candidates_before = Vec::with_capacity(questions.len());
    let mut results_after = Vec::with_capacity(questions.len());
    let mut filtered_all = 0;
    for ((question, query_vector), used_query) in
        questions.iter().zip(query_vectors).zip(used_queries)
    {
        let (candidates, results) = retrieve(
            chunks,
            query_vector,
            used_query,
            SearchOptions {
                top_k: params.top_k,
                candidate_top_k: params.candidate_top_k,
                similarity_threshold: params.similarity_threshold,
                rerank: true,
            },
        );
        candidates_before.push(candidates);
        if params.similarity_threshold.is_some()
            && candidates_before.last().copied().unwrap_or_default() > 0
            && results.is_empty()
        {
            filtered_all += 1;
        }
        results_after.push(results.len());
        ranks.push(
            results
                .iter()
                .position(|&(_, index)| question.hits(&chunks[index]))
                .map(|rank| rank + 1),
        );
    }
    ModeEvaluation {
        strategy: strategy.to_string(),
        mode: match (params.rewrite, params.similarity_threshold.is_some()) {
            (false, false) => "baseline",
            (true, false) => "rewrite",
            (false, true) => "filter",
            (true, true) => "rewrite+filter",
        }
        .into(),
        params,
        ranks,
        candidates_before,
        results_after,
        filtered_all,
        rewrite_fallbacks,
    }
}

fn mode_params(e: &ModeEvaluation) -> String {
    format!(
        "candidate_top_k={}, top_k={}, threshold={}, rewrite={}",
        e.params.candidate_top_k,
        e.params.top_k,
        e.params
            .similarity_threshold
            .map_or_else(|| "off".into(), |value| format!("{value:.3}")),
        e.params.rewrite
    )
}

/// Добавляет сравнение четырёх режимов к существующему отчёту chunking.
pub fn render_modes(evals: &[ModeEvaluation], questions: &[Question]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n## Сравнение режимов RAG\n");
    let _ = writeln!(
        out,
        "Один набор вопросов и один индекс; hit@k использует top_k соответствующего режима."
    );
    let _ = writeln!(
        out,
        "| Режим | Стратегия | Параметры | hit@1 | hit@k | MRR | Без результата | Фильтр удалил всех | Среднее кандидатов | Среднее результатов | Fallback rewrite |"
    );
    let _ = writeln!(out, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    for e in evals {
        let n = questions.len();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {} | {} | {:.2} | {:.2} | {} |",
            e.mode,
            e.strategy,
            mode_params(e),
            e.hit_at(1),
            e.hit_at(e.params.top_k),
            e.mrr(),
            e.no_results(),
            e.filtered_all,
            e.average_candidates(),
            e.average_results(),
            e.rewrite_fallbacks.min(n),
        );
    }
    let _ = writeln!(out, "\n### Baseline\n");
    for name in ["baseline", "rewrite"] {
        for e in evals.iter().filter(|e| e.mode == name) {
            let _ = writeln!(
                out,
                "- {} / {}: hit@1 {:.3}, hit@k {:.3}, MRR {:.3}, без результата {}.",
                e.mode,
                e.strategy,
                e.hit_at(1),
                e.hit_at(e.params.top_k),
                e.mrr(),
                e.no_results()
            );
        }
    }
    let _ = writeln!(out, "\n### Improved RAG\n");
    for name in ["filter", "rewrite+filter"] {
        for e in evals.iter().filter(|e| e.mode == name) {
            let _ = writeln!(
                out,
                "- {} / {}: hit@1 {:.3}, hit@k {:.3}, MRR {:.3}, без результата {}, среднее кандидатов {:.2}, среднее результатов {:.2}.",
                e.mode,
                e.strategy,
                e.hit_at(1),
                e.hit_at(e.params.top_k),
                e.mrr(),
                e.no_results(),
                e.average_candidates(),
                e.average_results()
            );
        }
    }
    out
}

impl Evaluation {
    pub fn hit_at(&self, k: usize) -> f64 {
        ratio(
            self.ranks
                .iter()
                .filter(|r| r.is_some_and(|r| r <= k))
                .count(),
            self.ranks.len(),
        )
    }

    pub fn mrr(&self) -> f64 {
        let sum: f64 = self
            .ranks
            .iter()
            .map(|r| r.map_or(0.0, |r| 1.0 / r as f64))
            .sum();
        if self.ranks.is_empty() {
            0.0
        } else {
            sum / self.ranks.len() as f64
        }
    }
}

fn ratio(n: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        n as f64 / total as f64
    }
}

/// Базовый режим поиска с умолчаниями `index_search`, без порога и rewrite.
pub fn evaluate(
    strategy: &str,
    chunks: &[StoredChunk],
    build: Option<BuildInfo>,
    questions: &[Question],
    query_vectors: &[Vec<f32>],
) -> Evaluation {
    let lengths: Vec<usize> = chunks.iter().map(|c| c.row.text.chars().count()).collect();
    let ranks = questions
        .iter()
        .zip(query_vectors)
        .map(|(q, qv)| {
            let (_, scored) = retrieve(
                chunks,
                qv,
                &q.question,
                SearchOptions {
                    top_k: crate::search::DEFAULT_TOP_K,
                    candidate_top_k: crate::search::DEFAULT_CANDIDATE_TOP_K,
                    similarity_threshold: None,
                    rerank: true,
                },
            );
            scored
                .iter()
                .position(|&(_, i)| q.hits(&chunks[i]))
                .map(|p| p + 1)
        })
        .collect();
    let unreachable = questions
        .iter()
        .filter(|q| !chunks.iter().any(|c| q.hits(c)))
        .count();
    Evaluation {
        strategy: strategy.to_string(),
        build,
        chunks: chunks.len(),
        lengths: length_stats(&lengths),
        crosses: chunks.iter().filter(|c| c.row.crosses_section).count(),
        cut: chunks.iter().filter(|c| c.row.cut_mid_sentence).count(),
        ranks,
        unreachable,
    }
}

fn pct(n: usize, total: usize) -> String {
    format!("{:.1}% ({n})", 100.0 * ratio(n, total))
}

fn embed_time(e: &Evaluation) -> String {
    match &e.build {
        Some(b) if b.chunks > 0 => format!(
            "{:.1} с ({:.0} мс на чанк)",
            b.embed_ms as f64 / 1000.0,
            b.embed_ms as f64 / b.chunks as f64
        ),
        Some(b) => format!("{:.1} с", b.embed_ms as f64 / 1000.0),
        None => "—".into(),
    }
}

/// Отчёт в Markdown: сводная таблица, ранги по вопросам и вывод.
pub fn render(evals: &[Evaluation], questions: &[Question], model: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Сравнение стратегий chunking\n");
    let _ = writeln!(
        out,
        "Отчёт сгенерирован командой `index-mcp compare` по базе индекса; все числа — из прогона.\n"
    );
    if let Some(b) = evals.iter().find_map(|e| e.build.as_ref()) {
        let _ = writeln!(
            out,
            "- Корпус: {} файлов `.docx`, {} символов.\n- Модель эмбеддингов: `{model}`, размерность {}.\n- Вопросов: {}.",
            b.files,
            b.chars,
            b.dim,
            questions.len()
        );
    }
    for e in evals {
        if let Some(b) = &e.build {
            let _ = writeln!(out, "- Параметры `{}`: `{}`.", e.strategy, b.params);
        }
    }
    let _ = writeln!(out, "\n## Метрики\n");
    let header: Vec<String> = evals.iter().map(|e| format!("`{}`", e.strategy)).collect();
    let _ = writeln!(out, "| Метрика | {} |", header.join(" | "));
    let _ = writeln!(out, "|---|{}", "---|".repeat(evals.len()));
    let mut row = |name: &str, f: &dyn Fn(&Evaluation) -> String| {
        let cells: Vec<String> = evals.iter().map(f).collect();
        let _ = writeln!(out, "| {name} | {} |", cells.join(" | "));
    };
    row("Чанков", &|e| e.chunks.to_string());
    row(
        "Длина, символов: min / median / p95 / max",
        &|e| {
            let l = e.lengths;
            format!("{} / {} / {} / {}", l.min, l.median, l.p95, l.max)
        },
    );
    row("Пересекают границу раздела", &|e| {
        pct(e.crosses, e.chunks)
    });
    row("Обрезаны посреди предложения", &|e| pct(e.cut, e.chunks));
    row("Время эмбеддинга", &embed_time);
    let n = questions.len();
    row("hit@1", &|e| {
        format!(
            "{:.2} ({}/{n})",
            e.hit_at(1),
            (e.hit_at(1) * n as f64).round()
        )
    });
    row("hit@5", &|e| {
        format!(
            "{:.2} ({}/{n})",
            e.hit_at(5),
            (e.hit_at(5) * n as f64).round()
        )
    });
    row("MRR", &|e| format!("{:.3}", e.mrr()));
    row(
        "Ожидаемого раздела нет среди чанков",
        &|e| format!("{} из {n}", e.unreachable),
    );

    let _ = writeln!(out, "\n## Ранг первого попадания по вопросам\n");
    let _ = writeln!(
        out,
        "| # | Вопрос | Ожидаемый раздел | {} |",
        header.join(" | ")
    );
    let _ = writeln!(out, "|---|---|---|{}", "---|".repeat(evals.len()));
    for (i, q) in questions.iter().enumerate() {
        let ranks: Vec<String> = evals
            .iter()
            .map(|e| e.ranks[i].map_or("—".into(), |r| r.to_string()))
            .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            i + 1,
            q.question.replace('|', "\\|"),
            q.expected_section.display().replace('|', "\\|"),
            ranks.join(" | ")
        );
    }
    let _ = writeln!(
        out,
        "\n«—» — чанка с ожидаемым разделом нет во всей выдаче."
    );
    let _ = writeln!(out, "\n## Вывод\n\n{}", conclusion(evals, model));
    out
}

fn list<T>(evals: &[Evaluation], f: impl Fn(&Evaluation) -> T) -> String
where
    T: std::fmt::Display,
{
    evals
        .iter()
        .map(|e| format!("{} у `{}`", f(e), e.strategy))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Вывод из цифр по фиксированным правилам: текст не сочиняется, а
/// собирается из тех же чисел, что в таблице, поэтому не расходится с ней.
pub fn conclusion(evals: &[Evaluation], model: &str) -> String {
    if evals.is_empty() {
        return "Нет построенных стратегий.".into();
    }
    let mut sentences = Vec::new();
    let best = evals
        .iter()
        .max_by(|a, b| {
            a.mrr()
                .total_cmp(&b.mrr())
                .then(a.hit_at(5).total_cmp(&b.hit_at(5)))
                .then(b.chunks.cmp(&a.chunks))
        })
        .expect("список не пуст");
    let tie = evals
        .iter()
        .filter(|e| e.strategy != best.strategy)
        .all(|e| {
            (e.mrr() - best.mrr()).abs() < 1e-9 && (e.hit_at(5) - best.hit_at(5)).abs() < 1e-9
        });
    if evals.len() > 1 && tie {
        sentences.push(format!(
            "По качеству поиска стратегии не различаются: MRR {:.3}, hit@5 {:.2}.",
            best.mrr(),
            best.hit_at(5)
        ));
    } else {
        sentences.push(format!(
            "По качеству поиска на этом корпусе лучше `{}`: MRR {}; hit@1 {}; hit@5 {}.",
            best.strategy,
            list(evals, |e| format!("{:.3}", e.mrr())),
            list(evals, |e| format!("{:.2}", e.hit_at(1))),
            list(evals, |e| format!("{:.2}", e.hit_at(5))),
        ));
    }
    sentences.push(format!(
        "Границу раздела пересекают {}, посреди предложения обрезаны {}.",
        list(evals, |e| format!(
            "{:.1}%",
            100.0 * ratio(e.crosses, e.chunks)
        )),
        list(evals, |e| format!("{:.1}%", 100.0 * ratio(e.cut, e.chunks))),
    ));
    sentences.push(format!(
        "Чанков {}, медианная длина {} символов, максимальная {}.",
        list(evals, |e| e.chunks),
        list(evals, |e| e.lengths.median),
        list(evals, |e| e.lengths.max),
    ));
    let unreachable: Vec<&Evaluation> = evals.iter().filter(|e| e.unreachable > 0).collect();
    if !unreachable.is_empty() {
        sentences.push(format!(
            "Часть вопросов попасть не могла: ожидаемого раздела нет среди чанков ({}) — раздел склеен с соседним или назван в вопросе иначе.",
            unreachable
                .iter()
                .map(|e| format!("{} у `{}`", e.unreachable, e.strategy))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    sentences.push(format!(
        "Абсолютные значения ограничены моделью `{model}`, обученной в основном на английском, при русских конспектах; сравнение при этом честное, потому что модель, префиксы и вопросы у стратегий общие."
    ));
    sentences.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ChunkRow;

    fn chunk(section: &str, vector: Vec<f32>, text_len: usize) -> StoredChunk {
        StoredChunk {
            row: ChunkRow {
                chunk_id: format!("s:{section}"),
                strategy: "s".into(),
                source: "dir/a.docx".into(),
                title: "A".into(),
                section: section.into(),
                ordinal: 0,
                char_start: 0,
                char_end: text_len as i64,
                text: "ж".repeat(text_len),
                crosses_section: section == "B",
                cut_mid_sentence: false,
            },
            model: "m".into(),
            vector,
        }
    }

    fn q(expected: &str) -> Question {
        Question {
            question: format!("про {expected}"),
            expected_section: Expected::One(expected.into()),
            source: None,
        }
    }

    #[test]
    fn ranks_hits_and_mrr() {
        let chunks = vec![
            chunk("A", vec![1.0, 0.0], 10),
            chunk("B", vec![0.0, 1.0], 20),
            chunk("C", vec![std::f32::consts::FRAC_1_SQRT_2; 2], 30),
        ];
        let questions = vec![q("A"), q("B"), q("Нет такого")];
        let qv = vec![vec![1.0, 0.0], vec![0.6, 0.8], vec![1.0, 0.0]];
        let e = evaluate("s", &chunks, None, &questions, &qv);
        // Вопрос B: C (0.99) выше B (0.8) → ранг 2.
        assert_eq!(e.ranks, vec![Some(1), Some(2), None]);
        assert!((e.hit_at(1) - 1.0 / 3.0).abs() < 1e-9);
        assert!((e.hit_at(5) - 2.0 / 3.0).abs() < 1e-9);
        assert!((e.mrr() - 0.5).abs() < 1e-9);
        assert_eq!(e.unreachable, 1);
        assert_eq!(e.crosses, 1);
        assert_eq!(
            e.lengths,
            LenStats {
                min: 10,
                median: 20,
                p95: 30,
                max: 30
            }
        );
    }

    #[test]
    fn evaluates_all_rag_modes_with_filter_metrics() {
        let chunks = vec![
            chunk("A", vec![1.0, 0.0], 10),
            chunk("B", vec![0.0, 1.0], 20),
        ];
        let questions = vec![q("A"), q("B")];
        let vectors = vec![vec![0.6, 0.0], vec![0.0, 0.6]];
        let rewritten_vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let modes = [
            (false, None, &vectors),
            (true, None, &rewritten_vectors),
            (false, Some(0.9), &vectors),
            (true, Some(0.9), &rewritten_vectors),
        ];
        let evaluations: Vec<_> = modes
            .into_iter()
            .map(|(rewrite, threshold, query_vectors)| {
                evaluate_mode(
                    "s",
                    &chunks,
                    &questions,
                    query_vectors,
                    &questions
                        .iter()
                        .map(|q| q.question.clone())
                        .collect::<Vec<_>>(),
                    ModeParams {
                        rewrite,
                        candidate_top_k: 2,
                        top_k: 1,
                        similarity_threshold: threshold,
                    },
                    0,
                )
            })
            .collect();
        assert_eq!(
            evaluations
                .iter()
                .map(|e| e.mode.as_str())
                .collect::<Vec<_>>(),
            ["baseline", "rewrite", "filter", "rewrite+filter"]
        );
        assert_eq!(evaluations[0].hit_at(1), 1.0);
        assert_eq!(evaluations[1].hit_at(1), 1.0);
        assert_eq!(evaluations[2].hit_at(1), 0.0);
        assert_eq!(evaluations[3].hit_at(1), 1.0);
        assert_eq!(evaluations[2].no_results(), 2);
        assert_eq!(evaluations[2].filtered_all, 2);
        assert_eq!(evaluations[2].average_results(), 0.0);
        assert_eq!(evaluations[3].no_results(), 0);
        assert!(render_modes(&evaluations, &questions).contains("Среднее кандидатов"));
    }

    #[test]
    fn compare_uses_the_search_ranking() {
        let mut chunks = vec![
            chunk("A", vec![1.0, 0.0], 10),
            chunk("B", vec![0.9, 0.0], 10),
        ];
        chunks[1].row.text = "Здесь точная фраза из вопроса".into();
        let question = Question {
            question: "Где „точная фраза“?".into(),
            expected_section: Expected::One("B".into()),
            source: None,
        };
        let queries = [question.question.clone()];
        let evaluation = evaluate_mode(
            "s",
            &chunks,
            std::slice::from_ref(&question),
            &[vec![1.0, 0.0]],
            &queries,
            ModeParams {
                rewrite: false,
                candidate_top_k: 2,
                top_k: 1,
                similarity_threshold: Some(0.5),
            },
            0,
        );
        assert_eq!(evaluation.ranks, vec![Some(1)]);
        assert_eq!(evaluation.results_after, vec![1]);
    }

    #[test]
    fn source_and_alternatives_narrow_the_match() {
        let c = chunk("A", vec![1.0], 1);
        let mut question = q("A");
        question.source = Some("a.docx".into());
        assert!(question.hits(&c));
        question.source = Some("b.docx".into());
        assert!(!question.hits(&c));
        let any = Question {
            question: String::new(),
            expected_section: Expected::Any(vec!["X".into(), "A".into()]),
            source: None,
        };
        assert!(any.hits(&c));
    }

    #[test]
    fn questions_json_accepts_string_or_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.json");
        std::fs::write(
            &path,
            r#"[{"question":"a","expected_section":"X > Y"},{"question":"b","expected_section":["X","Z"],"source":"f.docx"}]"#,
        )
        .unwrap();
        let qs = load_questions(&path).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[1].expected_section.display(), "X | Z");
        std::fs::write(&path, "[]").unwrap();
        assert!(load_questions(&path).is_err());
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let l: Vec<usize> = (1..=100).collect();
        assert_eq!(
            length_stats(&l),
            LenStats {
                min: 1,
                median: 50,
                p95: 95,
                max: 100
            }
        );
        assert_eq!(
            length_stats(&[7]),
            LenStats {
                min: 7,
                median: 7,
                p95: 7,
                max: 7
            }
        );
    }

    #[test]
    fn render_contains_table_and_conclusion() {
        let chunks = vec![chunk("A", vec![1.0, 0.0], 10)];
        let questions = vec![q("A")];
        let e1 = evaluate("fixed", &chunks, None, &questions, &[vec![1.0, 0.0]]);
        let e2 = evaluate("structure", &chunks, None, &questions, &[vec![0.0, 1.0]]);
        let md = render(&[e1, e2], &questions, "nomic-embed-text");
        assert!(md.contains("| Метрика | `fixed` | `structure` |"));
        assert!(md.contains("| MRR | 1.000 | 1.000 |"));
        assert!(md.contains("не различаются"), "{md}");
    }
}
