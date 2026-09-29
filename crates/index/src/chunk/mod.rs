//! Стратегии разбиения документа на чанки.
//!
//! Стратегия — реализация трейта [`Chunker`]: она видит разобранный
//! документ и возвращает чанки с текстом, разделом и смещениями. Индексатор,
//! эмбеддер и хранилище работают со списком `Box<dyn Chunker>` и о
//! конкретных стратегиях не знают; новая стратегия добавляется файлом рядом
//! и строкой в [`make`].

mod fixed;
mod structure;
pub mod units;

use crate::docx::{Document, SECTION_SEPARATOR};
use anyhow::{Result, bail};
use std::sync::Arc;
use units::Units;

pub use fixed::FixedChunker;
pub use structure::StructureChunker;

/// Чанк до присвоения `chunk_id`: его назначает индексатор по порядку.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub section: String,
    /// Символьные смещения фрагмента документа, из которого взят чанк.
    pub char_start: usize,
    pub char_end: usize,
}

pub trait Chunker: Send + Sync {
    /// Имя стратегии — значение колонки `strategy` и префикс `chunk_id`.
    fn name(&self) -> &'static str;
    /// Параметры стратегии для отчёта: с какими размерами построен индекс.
    fn params(&self) -> serde_json::Value;
    fn chunk(&self, doc: &Document) -> Vec<Chunk>;
}

/// Размеры из флагов CLI; каждая стратегия берёт свои. Все размеры — в
/// единицах `units`.
#[derive(Clone)]
pub struct ChunkParams {
    pub chunk_size: usize,
    pub overlap: usize,
    pub max_section: usize,
    pub min_section: usize,
    pub units: Arc<dyn Units>,
}

/// Имена всех стратегий в порядке вывода; `all` в CLI раскрывается в них.
pub const STRATEGIES: &[&str] = &["fixed", "structure"];

pub fn make(name: &str, params: &ChunkParams) -> Result<Box<dyn Chunker>> {
    Ok(match name {
        "fixed" => Box::new(FixedChunker::new(
            params.chunk_size,
            params.overlap,
            params.units.clone(),
        )?),
        "structure" => Box::new(StructureChunker::new(
            params.max_section,
            params.min_section,
            params.units.clone(),
        )?),
        other => bail!(
            "неизвестная стратегия {other:?}; есть: {}, all",
            STRATEGIES.join(", ")
        ),
    })
}

/// Стратегии по значению `--strategy`: имя или `all`.
pub fn select(spec: &str, params: &ChunkParams) -> Result<Vec<Box<dyn Chunker>>> {
    if spec == "all" {
        STRATEGIES.iter().map(|name| make(name, params)).collect()
    } else {
        Ok(vec![make(spec, params)?])
    }
}

/// Знаки, после которых предложение считается законченным. Двоеточие и
/// точка с запятой не входят: за ними в конспектах обычно идёт продолжение
/// мысли — список или пояснение.
pub fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Закрывающие кавычки и скобки, которые могут стоять после точки.
fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | '»' | '”' | ')' | ']')
}

/// Позиция сразу за концом предложения: `pos` стоит на пробельном символе,
/// а перед ним (через закрывающие кавычки) — знак конца предложения.
fn is_after_sentence(chars: &[char], pos: usize) -> bool {
    if pos == 0 || pos >= chars.len() || !chars[pos].is_whitespace() {
        return false;
    }
    let mut i = pos;
    while i > 0 && is_closer(chars[i - 1]) {
        i -= 1;
    }
    i > 0 && is_sentence_end(chars[i - 1])
}

/// Лучшая точка разреза в `[lo, hi]`: самый правый конец предложения или
/// абзаца, иначе самый правый пробел. Разрез в позиции `p` значит, что
/// чанк кончается перед символом `p`.
pub fn find_break(doc: &Document, lo: usize, hi: usize) -> Option<usize> {
    let chars = doc.chars();
    let hi = hi.min(chars.len());
    if lo >= hi {
        return None;
    }
    let sentence = (lo..=hi)
        .rev()
        .find(|&p| p == chars.len() || chars[p] == '\n' || is_after_sentence(chars, p));
    sentence.or_else(|| (lo..hi).rev().find(|&p| chars[p].is_whitespace()))
}

/// Начало предложения в `[lo, hi]`, ближайшее к `target`: позиция первого
/// символа после конца предложения или абзаца.
pub fn find_sentence_start(doc: &Document, lo: usize, hi: usize, target: usize) -> Option<usize> {
    let chars = doc.chars();
    let hi = hi.min(chars.len());
    (lo..hi)
        .filter(|&p| p > 0 && !chars[p].is_whitespace())
        .filter(|&p| chars[p - 1] == '\n' || is_after_sentence(chars, p - 1))
        .min_by_key(|&p| p.abs_diff(target))
}

/// Сужает `[start, end)` до непробельных краёв.
pub fn trim_range(doc: &Document, mut start: usize, mut end: usize) -> (usize, usize) {
    let chars = doc.chars();
    while start < end && chars[start].is_whitespace() {
        start += 1;
    }
    while end > start && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    (start, end)
}

/// Чанк пересекает границу раздела, если его абзацы не лежат в одном
/// поддереве разделов: среди их путей нет такого, чтобы остальные совпадали
/// с ним или были вложены в него. Вводный текст раздела вместе с его
/// подразделом границы не пересекает, соседние разделы или текст до
/// первого заголовка вместе с разделом — пересекают. Определение не
/// зависит от того, каким разделом стратегия подписала чанк.
pub fn crosses_section(doc: &Document, chunk: &Chunk) -> bool {
    let sections: Vec<&str> = doc
        .paragraphs
        .iter()
        .filter(|p| p.start < chunk.char_end && p.end > chunk.char_start)
        .map(|p| p.section.as_str())
        .collect();
    let Some(root) = sections.iter().min_by_key(|s| s.len()) else {
        return false;
    };
    let nested_prefix = format!("{root}{SECTION_SEPARATOR}");
    !sections
        .iter()
        .all(|s| s == root || (!root.is_empty() && s.starts_with(&nested_prefix)))
}

/// Чанк обрезан посреди предложения, если его конец не совпадает ни с
/// концом абзаца, ни со знаком конца предложения. Начало не проверяется:
/// у окон с перекрытием оно по построению бывает внутри предложения, и
/// метрика тогда мерила бы перекрытие, а не качество границ.
pub fn cut_mid_sentence(doc: &Document, chunk: &Chunk) -> bool {
    let chars = doc.chars();
    let end = chunk.char_end;
    if end == 0 || end >= chars.len() || chars[end] == '\n' {
        return false;
    }
    let mut i = end;
    while i > chunk.char_start && is_closer(chars[i - 1]) {
        i -= 1;
    }
    !(i > chunk.char_start && is_sentence_end(chars[i - 1]))
}

#[cfg(test)]
pub(crate) fn doc(raw: &[(&str, Option<u8>)]) -> Document {
    Document::from_paragraphs(
        "тест.docx".into(),
        "тест".into(),
        raw.iter().map(|(t, h)| (t.to_string(), *h)).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ChunkParams {
        ChunkParams {
            chunk_size: 1200,
            overlap: 200,
            max_section: 3000,
            min_section: 200,
            units: Arc::new(units::Chars),
        }
    }

    #[test]
    fn select_expands_all_and_rejects_unknown() {
        let names: Vec<_> = select("all", &params())
            .unwrap()
            .iter()
            .map(|c| c.name())
            .collect();
        assert_eq!(names, STRATEGIES);
        assert_eq!(select("fixed", &params()).unwrap()[0].name(), "fixed");
        assert!(select("semantic", &params()).is_err());
    }

    #[test]
    fn break_prefers_sentence_end_over_space() {
        let d = doc(&[("Первое предложение. Второе предложение идёт дальше", None)]);
        // «Первое предложение.» — 20 символов, пробел за точкой в позиции 19.
        assert_eq!(find_break(&d, 5, 40), Some(19));
        assert_eq!(find_break(&d, 25, 40), Some(38));
    }

    #[test]
    fn section_crossing_ignores_own_subsections() {
        let d = doc(&[
            ("Сети", Some(1)),
            ("Текст.", None),
            ("TCP", Some(2)),
            ("Текст.", None),
            ("ОС", Some(1)),
            ("Текст.", None),
        ]);
        let whole = |section: &str, start: usize, end: usize| Chunk {
            text: String::new(),
            section: section.into(),
            char_start: start,
            char_end: end,
        };
        let tcp_end = d.paragraphs[3].end;
        assert!(!crosses_section(&d, &whole("Сети", 0, tcp_end)));
        // Подпись чанка на ответ не влияет: важны только разделы его абзацев.
        assert!(!crosses_section(&d, &whole("Сети > TCP", 0, tcp_end)));
        // Конец TCP и начало ОС — соседние поддеревья.
        let os_start = d.paragraphs[4].start;
        assert!(crosses_section(
            &d,
            &whole("Сети > TCP", d.paragraphs[3].start, os_start + 2)
        ));
        assert!(!crosses_section(&d, &whole("ОС", os_start, d.char_len())));
        assert!(crosses_section(&d, &whole("Сети", 0, d.char_len())));
        assert!(crosses_section(
            &d,
            &whole("Сети > TCP", d.paragraphs[2].start, d.char_len())
        ));
    }

    #[test]
    fn mid_sentence_detection() {
        let d = doc(&[("Одно. Два три", None), ("Новый", None)]);
        let c = |end| Chunk {
            text: String::new(),
            section: String::new(),
            char_start: 0,
            char_end: end,
        };
        assert!(!cut_mid_sentence(&d, &c(5)));
        assert!(cut_mid_sentence(&d, &c(9)));
        assert!(!cut_mid_sentence(&d, &c(13)));
        assert!(!cut_mid_sentence(&d, &c(d.char_len())));
    }
}
