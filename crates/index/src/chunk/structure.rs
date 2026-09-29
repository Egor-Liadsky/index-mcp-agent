//! Чанки по структуре документа.
//!
//! Раздел — от заголовка до следующего заголовка того же или более высокого
//! уровня, вместе с подразделами. Раздел, который целиком помещается в
//! `max_section`, становится одним чанком. Более длинный режется по
//! границам абзацев: собственный текст раздела (заголовок и абзацы до
//! первого подраздела) — отдельными кусками, подразделы — рекурсивно тем же
//! правилом. Абзац длиннее `max_section` режется по концам предложений —
//! иначе один такой абзац дал бы чанк, который не влезет в контекст модели.
//!
//! Куски короче `min_section` (часто это заголовок с одной фразой перед
//! подразделами) склеиваются со следующим куском того же родителя, а
//! последний — с предыдущим; раздел склейки — раздел её большей части. Заголовок входит в текст чанка; продолжению
//! длинного раздела он дописывается первой строкой, чтобы кусок из середины
//! раздела не терял темы. Смещения чанка при этом указывают на фрагмент
//! документа без дописанного заголовка.

use super::{Chunk, Chunker, find_break, trim_range};
use crate::docx::Document;
use anyhow::{Result, ensure};
use std::ops::Range;

pub struct StructureChunker {
    max: usize,
    min: usize,
}

/// Узел дерева разделов: абзац-заголовок и граница его раздела.
struct Node {
    para: usize,
    end_para: usize,
    children: Vec<Node>,
}

/// Кусок текста до склейки: символьный диапазон, раздел и заголовок для
/// продолжения.
#[derive(Debug)]
struct Piece {
    start: usize,
    end: usize,
    section: String,
    prefix: Option<String>,
    /// Длина самой крупной составной части: её раздел и становится
    /// разделом склеенного куска.
    weight: usize,
}

impl Piece {
    fn new(start: usize, end: usize, section: String, prefix: Option<String>) -> Self {
        Self {
            start,
            end,
            section,
            prefix,
            weight: end - start,
        }
    }

    fn len(&self) -> usize {
        self.end - self.start
    }

    /// Дописывает следующий кусок. Раздел склейки — раздел большей части:
    /// заголовок раздела с одной вводной фразой, приклеенный к первому
    /// подразделу, не должен отнимать у подраздела его текст. Заголовок
    /// продолжения остаётся от первого куска — с него начинается текст.
    fn absorb(&mut self, next: Piece) {
        if next.weight > self.weight {
            self.section = next.section;
            self.weight = next.weight;
        }
        self.end = next.end;
    }
}

impl StructureChunker {
    pub fn new(max: usize, min: usize) -> Result<Self> {
        ensure!(
            max >= 20,
            "--max-section должен быть не меньше 20 символов, а не {max}"
        );
        ensure!(
            min < max,
            "--min-section ({min}) должен быть меньше --max-section ({max})"
        );
        Ok(Self { max, min })
    }

    fn node_pieces(&self, doc: &Document, node: &Node) -> Vec<Piece> {
        let paras = &doc.paragraphs;
        let heading = &paras[node.para];
        let span_end = paras[node.end_para - 1].end;
        if span_end - heading.start <= self.max {
            return vec![Piece::new(
                heading.start,
                span_end,
                heading.section.clone(),
                None,
            )];
        }
        let own_end = node.children.first().map_or(node.end_para, |c| c.para);
        let mut pieces = self.split_paragraphs(doc, node.para..own_end, Some(&heading.text));
        for child in &node.children {
            pieces.extend(self.node_pieces(doc, child));
        }
        self.merge(pieces)
    }

    /// Жадно набирает абзацы в куски не длиннее `max`.
    fn split_paragraphs(
        &self,
        doc: &Document,
        range: Range<usize>,
        heading: Option<&str>,
    ) -> Vec<Piece> {
        let paras = &doc.paragraphs[range];
        let Some(first) = paras.first() else {
            return Vec::new();
        };
        let section = first.section.clone();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut current: Option<(usize, usize)> = None;
        for p in paras {
            if p.end - p.start > self.max {
                // Длинный абзац продолжает набранный кусок и режется по предложениям.
                let mut s = current.take().map_or(p.start, |(s, _)| s);
                while p.end - s > self.max {
                    let cut =
                        find_break(doc, s + self.max / 2, s + self.max).unwrap_or(s + self.max);
                    spans.push(trim_range(doc, s, cut));
                    s = trim_range(doc, cut, p.end).0;
                }
                current = Some((s, p.end));
                continue;
            }
            if let Some((s, e)) = current {
                if p.end - s <= self.max {
                    current = Some((s, p.end));
                    continue;
                }
                spans.push((s, e));
            }
            current = Some((p.start, p.end));
        }
        spans.extend(current);
        spans
            .into_iter()
            .enumerate()
            .map(|(i, (start, end))| {
                let prefix = if i > 0 {
                    heading.map(str::to_string)
                } else {
                    None
                };
                Piece::new(start, end, section.clone(), prefix)
            })
            .collect()
    }

    /// Склеивает короткие куски одного родителя со следующими.
    fn merge(&self, pieces: Vec<Piece>) -> Vec<Piece> {
        let mut out: Vec<Piece> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match out.last_mut() {
                Some(last) if last.len() < self.min => last.absorb(piece),
                _ => out.push(piece),
            }
        }
        if out.len() >= 2 && out.last().is_some_and(|p| p.len() < self.min) {
            let tail = out.pop().expect("проверено выше");
            out.last_mut().expect("проверено выше").absorb(tail);
        }
        out
    }
}

/// Разделы уровня области `[from, to)`: каждый заголовок забирает абзацы до
/// следующего заголовка того же или более высокого уровня.
fn build_nodes(doc: &Document, from: usize, to: usize) -> Vec<Node> {
    let paras = &doc.paragraphs;
    let mut nodes = Vec::new();
    let mut i = from;
    while i < to {
        let Some(level) = paras[i].heading else {
            i += 1;
            continue;
        };
        let end = (i + 1..to)
            .find(|&j| paras[j].heading.is_some_and(|l| l <= level))
            .unwrap_or(to);
        nodes.push(Node {
            para: i,
            end_para: end,
            children: build_nodes(doc, i + 1, end),
        });
        i = end;
    }
    nodes
}

impl Chunker for StructureChunker {
    fn name(&self) -> &'static str {
        "structure"
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({ "max_section": self.max, "min_section": self.min })
    }

    fn chunk(&self, doc: &Document) -> Vec<Chunk> {
        let n = doc.paragraphs.len();
        let roots = build_nodes(doc, 0, n);
        let preamble_end = roots.first().map_or(n, |r| r.para);
        let mut pieces = self.split_paragraphs(doc, 0..preamble_end, None);
        for root in &roots {
            pieces.extend(self.node_pieces(doc, root));
        }
        self.merge(pieces)
            .into_iter()
            .map(|p| {
                let body = doc.slice(p.start, p.end);
                Chunk {
                    text: match &p.prefix {
                        Some(h) => format!("{h}\n{body}"),
                        None => body.to_string(),
                    },
                    section: p.section,
                    char_start: p.start,
                    char_end: p.end,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{crosses_section, doc};

    fn text(n: usize) -> String {
        "Предложение конспекта. ".repeat(n).trim().to_string()
    }

    #[test]
    fn fitting_sections_become_one_chunk_each_with_heading() {
        let body = text(12); // ≈ 280 символов
        let d = doc(&[
            ("Сети", Some(1)),
            (&body, None),
            ("ОС", Some(1)),
            (&body, None),
        ]);
        let chunks = StructureChunker::new(3000, 200).unwrap().chunk(&d);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("Сети\n"));
        assert_eq!(chunks[0].section, "Сети");
        assert_eq!(chunks[1].section, "ОС");
        assert!(chunks.iter().all(|c| !crosses_section(&d, c)));
    }

    #[test]
    fn section_fitting_whole_keeps_its_subsections() {
        let body = text(12);
        let d = doc(&[
            ("Сети", Some(1)),
            (&body, None),
            ("TCP", Some(2)),
            (&body, None),
        ]);
        let chunks = StructureChunker::new(3000, 200).unwrap().chunk(&d);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section, "Сети");
        assert!(!crosses_section(&d, &chunks[0]));
    }

    #[test]
    fn long_section_splits_at_paragraphs_and_repeats_heading() {
        let body = text(12);
        let mut raw = vec![("Сети", Some(1))];
        raw.extend(std::iter::repeat_n((body.as_str(), None), 8));
        let d = doc(&raw);
        let chunks = StructureChunker::new(1000, 200).unwrap().chunk(&d);
        assert!(chunks.len() >= 3);
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.char_end - c.char_start <= 1000, "{c:?}");
            assert!(c.text.starts_with("Сети"), "{i}: {:?}", c.text);
            assert_eq!(c.section, "Сети");
            // Кусок начинается с начала абзаца.
            assert!(d.paragraphs.iter().any(|p| p.start == c.char_start));
        }
    }

    #[test]
    fn short_intro_merges_with_first_subsection() {
        let body = text(40); // ≈ 920 символов
        let d = doc(&[
            ("Сети", Some(1)),
            ("Кратко.", None),
            ("TCP", Some(2)),
            (&body, None),
            ("UDP", Some(2)),
            (&body, None),
        ]);
        let chunks = StructureChunker::new(1000, 200).unwrap().chunk(&d);
        let sections: Vec<_> = chunks.iter().map(|c| c.section.as_str()).collect();
        assert_eq!(sections, ["Сети > TCP", "Сети > UDP"]);
        assert!(chunks[0].text.starts_with("Сети\nКратко.\nTCP\n"));
        assert!(!crosses_section(&d, &chunks[0]));
    }

    #[test]
    fn short_last_piece_merges_backwards_within_parent() {
        let body = text(43); // TCP ≈ 992 символа, раздел целиком — больше 1000
        let d = doc(&[
            ("Сети", Some(1)),
            ("TCP", Some(2)),
            (&body, None),
            ("UDP", Some(2)),
            ("Коротко.", None),
        ]);
        let chunks = StructureChunker::new(1000, 200).unwrap().chunk(&d);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].char_end, d.char_len());
        assert!(chunks[0].char_end - chunks[0].char_start > 1000);
    }

    #[test]
    fn oversized_paragraph_is_cut_at_sentences() {
        let d = doc(&[("Сети", Some(1)), (&text(200), None)]);
        let chunks = StructureChunker::new(1000, 200).unwrap().chunk(&d);
        assert!(chunks.len() >= 4);
        for c in &chunks {
            // Короткий хвост абзаца приклеивается к предыдущему куску.
            assert!(c.char_end - c.char_start <= 1000 + 200);
            assert!(c.text.ends_with('.'), "{:?}", &c.text[c.text.len() - 20..]);
        }
    }

    #[test]
    fn preamble_before_first_heading_has_empty_section() {
        let body = text(12);
        let d = doc(&[(&body, None), ("Сети", Some(1)), (&body, None)]);
        let chunks = StructureChunker::new(3000, 200).unwrap().chunk(&d);
        assert_eq!(chunks[0].section, "");
        assert_eq!(chunks[1].section, "Сети");
    }

    #[test]
    fn document_without_headings_still_chunks() {
        let body = text(12);
        let d = doc(&[(&body, None), (&body, None), (&body, None)]);
        let chunks = StructureChunker::new(600, 200).unwrap().chunk(&d);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.section.is_empty()));
    }
}
