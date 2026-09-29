//! Счёт токенов модели эмбеддингов — для размеров чанков в токенах.
//!
//! `nomic-embed-text` — BERT с токенизатором WordPiece. Словарь берётся у
//! самого Ollama (`POST /api/show` с `verbose: true`, поле
//! `tokenizer.ggml.tokens`), а не скачивается отдельно: так словарь всегда
//! тот, которым модель режет вход, и внешних источников, кроме Ollama, нет.
//!
//! Алгоритм повторяет WPM-токенизатор llama.cpp, на котором Ollama считает
//! BERT-модели: NFD с удалением диакритики (`й` → `и`), нижний регистр,
//! разбиение по пробелам, пунктуация и ASCII-символы — отдельные слова;
//! слово с префиксом `▁` режется жадно самыми длинными токенами словаря,
//! слово без разбора — один `[UNK]`. Счёт не обязан совпасть до токена:
//! переполнение контекста всё равно ловит `truncate: false`, а живой тест
//! сверяет счёт с `prompt_eval_count` Ollama.

use crate::chunk::units::{Measure, Units};
use crate::docx::Document;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// «Фантомный пробел», которым GGUF-словарь помечает токены начала слова.
const PHANTOM: char = '\u{2581}';

pub struct WordPiece {
    vocab: HashSet<String>,
    /// Длина самого длинного токена в символах — предел жадного поиска.
    max_len: usize,
}

impl WordPiece {
    /// Словарь в формате GGUF (`▁слово` — начало слова, `часть` —
    /// продолжение) или исходном BERT (`слово` и `##часть`); второй
    /// приводится к первому.
    pub fn from_tokens(tokens: Vec<String>) -> Result<Self> {
        ensure!(!tokens.is_empty(), "пустой словарь токенизатора");
        let bert_format = tokens.iter().any(|t| t.starts_with("##"));
        let vocab: HashSet<String> = tokens
            .into_iter()
            .map(|t| {
                if !bert_format || (t.starts_with('[') && t.ends_with(']')) {
                    t
                } else if let Some(rest) = t.strip_prefix("##") {
                    rest.to_string()
                } else {
                    format!("{PHANTOM}{t}")
                }
            })
            .collect();
        let max_len = vocab.iter().map(|t| t.chars().count()).max().unwrap_or(1);
        Ok(Self { vocab, max_len })
    }

    /// Словарь модели у Ollama.
    pub async fn from_ollama(url: &str, model: &str) -> Result<Self> {
        let endpoint = format!("{}/api/show", url.trim_end_matches('/'));
        let response = reqwest::Client::new()
            .post(&endpoint)
            .json(&json!({ "model": model, "verbose": true }))
            .send()
            .await
            .with_context(|| format!("Ollama недоступен по {endpoint}"))?;
        let status = response.status();
        let text = response.text().await.context("не прочитать ответ Ollama")?;
        if !status.is_success() {
            bail!("Ollama ответил {status} на /api/show: {}", text.trim());
        }
        let body: Value = serde_json::from_str(&text).context("неожиданный ответ /api/show")?;
        let info = &body["model_info"];
        let kind = info["tokenizer.ggml.model"]
            .as_str()
            .unwrap_or("неизвестный");
        ensure!(
            kind == "bert",
            "токенизатор модели {model} — {kind}; размер в токенах умеем считать только для WordPiece (bert)"
        );
        let tokens: Vec<String> = info["tokenizer.ggml.tokens"]
            .as_array()
            .filter(|a| !a.is_empty())
            .with_context(|| {
                format!("Ollama не вернул словарь модели {model} (tokenizer.ggml.tokens)")
            })?
            .iter()
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect();
        Self::from_tokens(tokens)
    }

    /// Слова текста с символьной позицией начала каждого.
    fn words(&self, text: &str) -> Vec<(usize, String)> {
        let mut words = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        for (i, c) in text.chars().enumerate() {
            for x in std::iter::once(c).nfd().filter(|x| !is_combining_mark(*x)) {
                if x.is_whitespace() {
                    if !current.is_empty() {
                        words.push((current_start, std::mem::take(&mut current)));
                    }
                    continue;
                }
                if x == '\0' || x == '\u{FFFD}' || x.is_control() {
                    continue;
                }
                let lower: String = x.to_lowercase().collect();
                if is_punctuation(x) || is_cjk(x) {
                    if !current.is_empty() {
                        words.push((current_start, std::mem::take(&mut current)));
                    }
                    words.push((i, lower));
                    continue;
                }
                if current.is_empty() {
                    current_start = i;
                }
                current.push_str(&lower);
            }
        }
        if !current.is_empty() {
            words.push((current_start, current));
        }
        words
    }

    /// Токенов в слове: жадно самые длинные куски словаря после `▁`.
    fn word_tokens(&self, word: &str) -> usize {
        let chars: Vec<char> = std::iter::once(PHANTOM).chain(word.chars()).collect();
        let n = chars.len();
        let mut count = 0;
        let mut i = 0;
        let mut piece = String::new();
        while i < n {
            let found = (i + 1..=(i + self.max_len).min(n)).rev().find(|&j| {
                piece.clear();
                piece.extend(&chars[i..j]);
                self.vocab.contains(&piece)
            });
            match found {
                Some(j) => {
                    count += 1;
                    i = j;
                }
                // Как в llama.cpp: слово без полного разбора — один [UNK].
                None => return 1,
            }
        }
        count
    }

    /// Токенов в тексте без служебных `[CLS]` и `[SEP]`.
    pub fn count(&self, text: &str) -> usize {
        self.words(text)
            .iter()
            .map(|(_, w)| self.word_tokens(w))
            .sum()
    }
}

/// Пунктуация по классу Unicode P* (приближённо: ASCII, общая пунктуация,
/// кавычки-ёлочки, CJK) и ASCII-символы — так их выделяет BERT.
fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(c, '«' | '»' | '¡' | '¿' | '§' | '¶' | '·')
        || ('\u{2010}'..='\u{2027}').contains(&c)
        || ('\u{2030}'..='\u{205E}').contains(&c)
        || ('\u{3000}'..='\u{303F}').contains(&c)
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// Длина в токенах модели.
pub struct TokenUnits(pub WordPiece);

impl Units for TokenUnits {
    fn name(&self) -> &'static str {
        "tokens"
    }

    fn measure(&self, doc: &Document) -> Measure {
        let mut at = vec![0usize; doc.char_len() + 1];
        for (start, word) in self.0.words(&doc.text) {
            at[start + 1] += self.0.word_tokens(&word);
        }
        for i in 1..at.len() {
            at[i] += at[i - 1];
        }
        Measure::new(at)
    }

    fn count(&self, text: &str) -> usize {
        self.0.count(text)
    }
}

#[cfg(test)]
pub mod testutil {
    /// Маленький словарь в формате GGUF: хватает на тексты тестов, остальное — [UNK].
    pub fn vocab() -> Vec<String> {
        [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "▁при",
            "вет",
            "▁мир",
            "▁,",
            "▁.",
            "▁иод",
            "▁сети",
            "▁tcp",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn wp() -> WordPiece {
        WordPiece::from_tokens(testutil::vocab()).unwrap()
    }

    #[test]
    fn greedy_longest_match_with_punctuation() {
        // привет → ▁при + вет; «,» и «.» — отдельные слова.
        assert_eq!(wp().count("Привет, мир."), 5);
        assert_eq!(wp().count("  мир  мир "), 2);
    }

    #[test]
    fn accents_are_stripped_and_unknown_words_are_one_token() {
        assert_eq!(wp().count("Йод"), 1);
        assert_eq!(wp().count("абракадабра"), 1);
        // Неразобранный хвост делает всё слово одним [UNK].
        assert_eq!(wp().count("приветы"), 1);
    }

    #[test]
    fn bert_format_is_converted() {
        let wp =
            WordPiece::from_tokens(vec!["[UNK]".into(), "при".into(), "##вет".into()]).unwrap();
        assert_eq!(wp.count("привет"), 2);
        assert_eq!(wp.count("вет"), 1); // «вет» в начале слова — не токен, значит [UNK]
    }

    #[test]
    fn measure_attributes_tokens_to_word_starts() {
        let d = crate::chunk::doc(&[("Привет, мир.", None)]);
        let m = TokenUnits(wp()).measure(&d);
        assert_eq!(m.len(0, d.char_len()), 5);
        assert_eq!(m.len(0, 6), 2); // «Привет»
        assert_eq!(m.len(6, 8), 1); // «,» и пробел
        assert_eq!(m.advance(0, 3), 8);
    }

    #[tokio::test]
    async fn vocabulary_comes_from_ollama_show() {
        let server = MockServer::start().await;
        Mock::given(path("/api/show"))
            .and(body_partial_json(
                json!({ "model": "nomic-embed-text", "verbose": true }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model_info": {
                    "tokenizer.ggml.model": "bert",
                    "tokenizer.ggml.tokens": testutil::vocab(),
                }
            })))
            .mount(&server)
            .await;
        let wp = WordPiece::from_ollama(&server.uri(), "nomic-embed-text")
            .await
            .unwrap();
        assert_eq!(wp.count("Привет, мир."), 5);
    }

    #[tokio::test]
    async fn non_wordpiece_model_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(path("/api/show"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model_info": { "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": ["a"] }
            })))
            .mount(&server)
            .await;
        let err = WordPiece::from_ollama(&server.uri(), "llama")
            .await
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("gpt2"), "{err}");
    }
}
