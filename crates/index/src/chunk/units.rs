//! Единицы длины чанка: символы или токены модели эмбеддингов.
//!
//! Стратегии режут текст по символьным позициям документа, а размер
//! окна, перекрытие и потолок раздела меряют в единицах. [`Measure`] —
//! накопленное число единиц по позициям, поэтому длина любого фрагмента и
//! «сдвиг на N единиц» считаются за O(1) и O(log n) без повторной
//! токенизации каждого кандидата на разрез.

use crate::docx::Document;

/// Способ считать длину. Реализации: [`Chars`] и токенизатор модели
/// (`tokens::TokenUnits`).
pub trait Units: Send + Sync {
    /// Имя единицы для отчёта и сообщений: `chars` или `tokens`.
    fn name(&self) -> &'static str;
    /// Разметка документа: сколько единиц до каждой символьной позиции.
    fn measure(&self, doc: &Document) -> Measure;
    /// Длина отдельного текста — дописанного заголовка или префикса задачи.
    fn count(&self, text: &str) -> usize;
}

/// `cum[i]` — число единиц в тексте документа до символа `i`; неубывает.
/// Токен приписывается позиции начала своего слова, поэтому длина
/// фрагмента, границы которого стоят между словами, точна.
#[derive(Debug, Clone)]
pub struct Measure {
    cum: Vec<usize>,
}

impl Measure {
    pub fn new(cum: Vec<usize>) -> Self {
        debug_assert!(cum.windows(2).all(|w| w[0] <= w[1]));
        Self { cum }
    }

    /// Единиц в `[start, end)`.
    pub fn len(&self, start: usize, end: usize) -> usize {
        self.cum[end] - self.cum[start]
    }

    /// Самая дальняя позиция `p ≥ start`, для которой `len(start, p) ≤ n`.
    pub fn advance(&self, start: usize, n: usize) -> usize {
        let limit = self.cum[start] + n;
        let p = self.cum.partition_point(|&c| c <= limit);
        (p - 1).max(start)
    }

    /// Самая ранняя позиция `p ≤ end`, для которой `len(p, end) ≤ n`.
    pub fn retreat(&self, end: usize, n: usize) -> usize {
        let limit = self.cum[end].saturating_sub(n);
        self.cum.partition_point(|&c| c < limit).min(end)
    }
}

/// Длина в символах — прежнее поведение и умолчание CLI.
pub struct Chars;

impl Units for Chars {
    fn name(&self) -> &'static str {
        "chars"
    }

    fn measure(&self, doc: &Document) -> Measure {
        Measure::new((0..=doc.char_len()).collect())
    }

    fn count(&self, text: &str) -> usize {
        text.chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_and_retreat_on_uneven_units() {
        // Позиции 0..=6; единицы приписаны позициям 0, 3 и 5 (начала «слов»).
        let m = Measure::new(vec![0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(m.len(0, 6), 3);
        assert_eq!(m.advance(0, 1), 3);
        assert_eq!(m.advance(0, 2), 5);
        assert_eq!(m.advance(0, 10), 6);
        assert_eq!(m.advance(4, 0), 5);
        assert_eq!(m.retreat(6, 1), 4);
        assert_eq!(m.retreat(6, 2), 1);
        assert_eq!(m.retreat(6, 10), 0);
    }

    #[test]
    fn chars_measure_is_identity() {
        let d = crate::chunk::doc(&[("абв", None), ("где", None)]);
        let m = Chars.measure(&d);
        assert_eq!(m.len(1, 5), 4);
        assert_eq!(m.advance(1, 3), 4);
        assert_eq!(m.retreat(5, 2), 3);
        assert_eq!(Chars.count("ёж"), 2);
    }
}
