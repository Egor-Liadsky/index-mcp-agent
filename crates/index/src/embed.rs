//! Эмбеддинги через нативный API Ollama `POST {url}/api/embed`.
//!
//! Нативный API, а не OpenAI-совместимый `/v1/embeddings`, выбран ради
//! двух параметров, которых у совместимого нет: `options.num_ctx` и
//! `truncate: false` — без него Ollama молча обрезает вход по контексту, и
//! вектор длинного чанка описывал бы только его начало. С `truncate: false`
//! слишком длинный вход — явная ошибка [`ContextOverflow`] с номером входа.
//! `num_ctx` не может превысить обученный контекст модели (2048 токенов у
//! `nomic-embed-text`): длину чанков ограничивают стратегии, а не он.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub url: String,
    pub model: String,
    pub batch: usize,
    pub num_ctx: u32,
    /// Ожидаемая размерность вектора; другая — ошибка, а не молчаливая смесь.
    pub dim: usize,
}

pub struct Embedder {
    client: reqwest::Client,
    cfg: EmbedConfig,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<String>,
    truncate: bool,
    options: EmbedOptions,
}

#[derive(Serialize)]
struct EmbedOptions {
    num_ctx: u32,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl Embedder {
    pub fn new(cfg: EmbedConfig) -> Result<Self> {
        ensure!(cfg.batch > 0, "--batch должен быть больше нуля");
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("не создать HTTP-клиент")?;
        Ok(Self { client, cfg })
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Векторы текстов с префиксом задачи, L2-нормализованные, в порядке входа.
    /// Префикс нужен только модели: в хранилище текст пишется без него.
    pub async fn embed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for (batch_no, batch) in texts.chunks(self.cfg.batch).enumerate() {
            let input: Vec<String> = batch.iter().map(|t| format!("{prefix}{t}")).collect();
            let vectors = match self.request(input.clone()).await {
                Err(RequestError::ContextOverflow(detail)) => {
                    let offset = batch_no * self.cfg.batch;
                    return Err(self.find_overflow(&input, offset, detail).await);
                }
                other => other?,
            };
            ensure!(
                vectors.len() == batch.len(),
                "Ollama вернул {} векторов на {} входов",
                vectors.len(),
                batch.len()
            );
            for v in vectors {
                ensure!(
                    v.len() == self.cfg.dim,
                    "размерность вектора {} вместо ожидаемой {} (модель {})",
                    v.len(),
                    self.cfg.dim,
                    self.cfg.model
                );
                out.push(normalize(v)?);
            }
        }
        Ok(out)
    }

    /// Длина вектора, которую модель отдаёт на самом деле: по одному
    /// короткому входу, без сверки с `cfg.dim`. Запасной путь, когда
    /// `/api/show` не сообщил `embedding_length`.
    pub async fn probe_dim(&self) -> Result<usize> {
        let vectors = self.request(vec!["dimension probe".to_string()]).await?;
        vectors
            .first()
            .map(Vec::len)
            .context("Ollama не вернул вектор на пробный вход")
    }

    /// Ollama отвечает на переполнение контекста одной ошибкой на весь батч.
    /// Входы батча пересылаются по одному, чтобы назвать виновника: индекс
    /// нужен индексатору, чтобы указать `chunk_id`, а длина — чтобы подобрать
    /// `--max-section` или `--chunk-size`.
    async fn find_overflow(
        &self,
        input: &[String],
        offset: usize,
        detail: String,
    ) -> anyhow::Error {
        for (i, text) in input.iter().enumerate() {
            if let Err(RequestError::ContextOverflow(detail)) =
                self.request(vec![text.clone()]).await
            {
                return ContextOverflow {
                    index: offset + i,
                    chars: text.chars().count(),
                    model: self.cfg.model.clone(),
                    num_ctx: self.cfg.num_ctx,
                    detail,
                }
                .into();
            }
        }
        anyhow::anyhow!("Ollama: {detail} (по одному входы батча помещаются)")
    }

    async fn request(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, RequestError> {
        let url = format!("{}/api/embed", self.cfg.url.trim_end_matches('/'));
        let body = EmbedRequest {
            model: &self.cfg.model,
            input,
            truncate: false,
            options: EmbedOptions {
                num_ctx: self.cfg.num_ctx,
            },
        };
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Ollama недоступен по {url}"))?;
        let status = response.status();
        let text = response.text().await.context("не прочитать ответ Ollama")?;
        if !status.is_success() {
            // Текст ошибки — единственный признак: отдельного кода у Ollama нет.
            if text.contains("exceeds the context length") {
                return Err(RequestError::ContextOverflow(text.trim().to_string()));
            }
            return Err(anyhow::anyhow!("Ollama ответил {status}: {}", text.trim()).into());
        }
        let parsed: EmbedResponse = serde_json::from_str(&text)
            .with_context(|| format!("неожиданный ответ Ollama: {}", preview(&text)))?;
        Ok(parsed.embeddings)
    }
}

enum RequestError {
    ContextOverflow(String),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for RequestError {
    fn from(err: anyhow::Error) -> Self {
        RequestError::Other(err)
    }
}

impl From<RequestError> for anyhow::Error {
    fn from(err: RequestError) -> Self {
        match err {
            RequestError::ContextOverflow(detail) => anyhow::anyhow!("Ollama: {detail}"),
            RequestError::Other(err) => err,
        }
    }
}

/// Вход не помещается в контекст модели. Лежит внутри `anyhow::Error`,
/// индексатор достаёт его `downcast_ref` и добавляет `chunk_id`.
#[derive(Debug)]
pub struct ContextOverflow {
    /// Индекс входа в срезе, переданном в [`Embedder::embed`].
    pub index: usize,
    /// Длина входа в символах вместе с префиксом задачи.
    pub chars: usize,
    pub model: String,
    pub num_ctx: u32,
    pub detail: String,
}

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "вход №{} ({} символов) не помещается в контекст модели {} (num_ctx {}): {}",
            self.index, self.chars, self.model, self.num_ctx, self.detail
        )
    }
}

impl std::error::Error for ContextOverflow {}

fn preview(text: &str) -> String {
    text.chars().take(200).collect()
}

/// L2-нормализация: после неё косинус — скалярное произведение.
pub fn normalize(mut v: Vec<f32>) -> Result<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "нулевой или нечисловой вектор"
    );
    for x in &mut v {
        *x /= norm;
    }
    Ok(v)
}

#[cfg(test)]
pub mod testutil {
    //! Поддельный `/api/embed`: вектор — «мешок слов», разложенный хэшем по
    //! измерениям. Тексты с общими словами получаются близкими, поэтому
    //! сравнение стратегий в тестах осмысленно и без настоящей модели.

    use serde_json::{Value, json};
    use wiremock::{Request, Respond, ResponseTemplate};

    pub struct BagOfWords {
        pub dim: usize,
    }

    pub fn vector(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() > 2)
        {
            let word = word.to_lowercase();
            let h = word.bytes().fold(1469598103934665603u64, |h, b| {
                (h ^ b as u64).wrapping_mul(1099511628211)
            });
            v[(h % dim as u64) as usize] += 1.0;
        }
        // Постоянная составляющая, чтобы вектор без слов не был нулевым.
        v[0] += 0.01;
        v
    }

    impl Respond for BagOfWords {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let embeddings: Vec<Vec<f32>> = body["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| vector(t.as_str().unwrap(), self.dim))
                .collect();
            ResponseTemplate::new(200)
                .set_body_json(json!({ "model": body["model"], "embeddings": embeddings }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::BagOfWords;
    use super::*;
    use serde_json::{Value, json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(url: &str, batch: usize, dim: usize) -> EmbedConfig {
        EmbedConfig {
            url: url.into(),
            model: "nomic-embed-text".into(),
            batch,
            num_ctx: 8192,
            dim,
        }
    }

    #[tokio::test]
    async fn batches_prefix_and_options_reach_ollama() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(BagOfWords { dim: 8 })
            .expect(3)
            .mount(&server)
            .await;
        let e = Embedder::new(cfg(&server.uri(), 2, 8)).unwrap();
        let texts = ["один текст", "второй текст", "третий", "четвёртый", "пятый"];
        let vectors = e.embed("search_document: ", &texts).await.unwrap();
        assert_eq!(vectors.len(), 5);
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum();
            assert!((norm - 1.0).abs() < 1e-5);
        }
        let requests = server.received_requests().await.unwrap();
        let first: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(first["model"], "nomic-embed-text");
        assert_eq!(
            first["input"],
            json!([
                "search_document: один текст",
                "search_document: второй текст"
            ])
        );
        assert_eq!(first["options"]["num_ctx"], 8192);
        assert_eq!(first["truncate"], false);
    }

    #[tokio::test]
    async fn count_mismatch_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "embeddings": [[1.0, 0.0]] })),
            )
            .mount(&server)
            .await;
        let e = Embedder::new(cfg(&server.uri(), 32, 2)).unwrap();
        let err = e.embed("", &["а", "б"]).await.unwrap_err().to_string();
        assert!(err.contains("1 векторов на 2 входов"), "{err}");
    }

    #[tokio::test]
    async fn wrong_dimension_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "embeddings": [[1.0, 0.0, 0.0]] })),
            )
            .mount(&server)
            .await;
        let e = Embedder::new(cfg(&server.uri(), 32, 768)).unwrap();
        let err = e.embed("", &["а"]).await.unwrap_err().to_string();
        assert!(
            err.contains("размерность вектора 3 вместо ожидаемой 768"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn ollama_error_body_is_reported() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(ResponseTemplate::new(404).set_body_json(
                json!({ "error": "model \"nomic-embed-text\" not found, try pulling it first" }),
            ))
            .mount(&server)
            .await;
        let e = Embedder::new(cfg(&server.uri(), 32, 768)).unwrap();
        let err = e.embed("", &["а"]).await.unwrap_err().to_string();
        assert!(
            err.contains("404") && err.contains("try pulling it first"),
            "{err}"
        );
    }

    /// Ollama с контекстом в `limit` символов: длиннее — 400 на весь батч.
    struct SmallContext {
        limit: usize,
    }

    impl wiremock::Respond for SmallContext {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let inputs = body["input"].as_array().unwrap();
            if inputs
                .iter()
                .any(|t| t.as_str().unwrap().chars().count() > self.limit)
            {
                return ResponseTemplate::new(400).set_body_json(
                    json!({ "error": "the input length exceeds the context length" }),
                );
            }
            BagOfWords { dim: 4 }.respond(request)
        }
    }

    #[tokio::test]
    async fn context_overflow_names_the_input() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(SmallContext { limit: 10 })
            .mount(&server)
            .await;
        let e = Embedder::new(cfg(&server.uri(), 2, 4)).unwrap();
        let err = e
            .embed("p: ", &["один", "два", "три", "слишком длинный", "пять"])
            .await
            .unwrap_err();
        let overflow = err
            .downcast_ref::<ContextOverflow>()
            .expect("типизированная ошибка");
        assert_eq!(overflow.index, 3);
        assert_eq!(overflow.chars, "p: слишком длинный".chars().count());
        assert!(
            err.to_string().contains("exceeds the context length"),
            "{err}"
        );
    }

    #[test]
    fn zero_vector_is_rejected() {
        assert!(normalize(vec![0.0, 0.0]).is_err());
        assert_eq!(normalize(vec![3.0, 4.0]).unwrap(), vec![0.6, 0.8]);
    }
}
