//! Окна фиксированного размера с перекрытием.
//!
//! Граница окна не режет слово: конец ищется в последней четверти окна —
//! сначала конец предложения или абзаца, затем пробел; начало следующего
//! окна — начало предложения рядом с точкой «конец минус перекрытие», иначе
//! начало слова. Структуру документа стратегия не видит, раздел чанка —
//! последний заголовок перед его началом.

use super::{Chunk, Chunker, find_break, find_sentence_start, trim_range};
use crate::docx::Document;
use anyhow::{Result, ensure};

pub struct FixedChunker {
    size: usize,
    overlap: usize,
}

impl FixedChunker {
    pub fn new(size: usize, overlap: usize) -> Result<Self> {
        ensure!(
            size >= 20,
            "--chunk-size должен быть не меньше 20 символов, а не {size}"
        );
        ensure!(
            overlap < size / 2,
            "--overlap ({overlap}) должен быть меньше половины --chunk-size ({size})"
        );
        Ok(Self { size, overlap })
    }

    fn window_end(&self, doc: &Document, start: usize) -> usize {
        let target = start + self.size;
        if target >= doc.char_len() {
            return doc.char_len();
        }
        let lo = target - self.size / 4;
        find_break(doc, lo, target).unwrap_or(target)
    }

    fn next_start(&self, doc: &Document, start: usize, end: usize) -> usize {
        let chars = doc.chars();
        let target = end.saturating_sub(self.overlap).max(start + 1);
        let slack = self.overlap / 2;
        let lo = target.saturating_sub(slack).max(start + 1);
        let next =
            find_sentence_start(doc, lo, (target + slack).min(end), target).unwrap_or_else(|| {
                // Начало слова: первый непробельный символ после пробела справа от target.
                (target..end)
                    .find(|&p| p > 0 && chars[p - 1].is_whitespace() && !chars[p].is_whitespace())
                    .unwrap_or(end)
            });
        next.max(start + 1)
    }
}

impl Chunker for FixedChunker {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({ "chunk_size": self.size, "overlap": self.overlap })
    }

    fn chunk(&self, doc: &Document) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < doc.char_len() {
            let end = self.window_end(doc, start);
            let (s, e) = trim_range(doc, start, end);
            if s < e {
                chunks.push(Chunk {
                    text: doc.slice(s, e).to_string(),
                    section: doc.section_at(s).to_string(),
                    char_start: s,
                    char_end: e,
                });
            }
            if end >= doc.char_len() {
                break;
            }
            start = self.next_start(doc, start, end);
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{cut_mid_sentence, doc};

    fn long_doc() -> Document {
        let sentence = "Протокол TCP гарантирует доставку и порядок байтов в потоке. ";
        let mut raw = vec![("Сети".to_string(), Some(1))];
        raw.push((sentence.repeat(30), None));
        raw.push(("TCP".to_string(), Some(2)));
        raw.push((sentence.repeat(30), None));
        Document::from_paragraphs("a.docx".into(), "a".into(), raw)
    }

    #[test]
    fn windows_respect_size_and_cover_document() {
        let d = long_doc();
        let chunks = FixedChunker::new(300, 50).unwrap().chunk(&d);
        assert!(chunks.len() > 5);
        let mut covered = 0;
        for c in &chunks {
            assert!(c.char_end - c.char_start <= 300, "{c:?}");
            assert!(c.char_start <= covered, "дыра перед {c:?}");
            assert_eq!(c.text, d.slice(c.char_start, c.char_end));
            covered = c.char_end;
        }
        assert_eq!(covered, d.char_len());
    }

    #[test]
    fn windows_overlap_and_end_on_sentences() {
        let d = long_doc();
        let chunks = FixedChunker::new(300, 50).unwrap().chunk(&d);
        for pair in chunks.windows(2) {
            assert!(
                pair[1].char_start < pair[0].char_end,
                "нет перекрытия: {pair:?}"
            );
        }
        assert!(chunks.iter().all(|c| !cut_mid_sentence(&d, c)));
    }

    #[test]
    fn section_is_last_heading_before_start() {
        let d = long_doc();
        let chunks = FixedChunker::new(300, 50).unwrap().chunk(&d);
        assert_eq!(chunks[0].section, "Сети");
        let tcp_start = d.paragraphs[2].start;
        for c in &chunks {
            let want = if c.char_start >= tcp_start {
                "Сети > TCP"
            } else {
                "Сети"
            };
            assert_eq!(c.section, want);
        }
    }

    #[test]
    fn falls_back_to_word_boundary_without_punctuation() {
        let d = doc(&[(&"слово ".repeat(100), None)]);
        let chunks = FixedChunker::new(100, 20).unwrap().chunk(&d);
        for c in &chunks {
            assert!(c.text.split(' ').all(|w| w == "слово"), "{c:?}");
        }
    }

    #[test]
    fn rejects_overlap_too_large() {
        assert!(FixedChunker::new(1200, 600).is_err());
        assert!(FixedChunker::new(1200, 200).is_ok());
    }

    #[test]
    fn deterministic() {
        let d = long_doc();
        let c = FixedChunker::new(300, 50).unwrap();
        assert_eq!(c.chunk(&d), c.chunk(&d));
    }
}
