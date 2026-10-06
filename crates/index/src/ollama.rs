//! Справочные запросы к Ollama: какие модели умеют эмбеддинги и какой
//! длины вектор у модели.
//!
//! `/api/show` зовётся без `verbose`: с ним Ollama отдаёт весь словарь
//! токенизатора (десятки тысяч строк), а здесь нужны только `capabilities` и
//! метаданные `model_info`. Словарь целиком берёт `tokens.rs`.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EmbeddingModel {
    pub name: String,
    /// Длина вектора из `<arch>.embedding_length`; `None`, если Ollama её не сообщил.
    pub dim: Option<usize>,
    /// Обученный контекст в токенах из `<arch>.context_length`.
    pub context_length: Option<u64>,
    /// Размер весов в байтах.
    pub size: Option<u64>,
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()
        .context("не создать HTTP-клиент")
}

async fn get_json(client: &reqwest::Client, url: &str, path: &str) -> Result<Value> {
    let endpoint = format!("{}{path}", url.trim_end_matches('/'));
    let response = client
        .get(&endpoint)
        .send()
        .await
        .with_context(|| format!("Ollama недоступен по {endpoint}"))?;
    decode(response, path).await
}

async fn show(client: &reqwest::Client, url: &str, model: &str) -> Result<Value> {
    let endpoint = format!("{}/api/show", url.trim_end_matches('/'));
    let response = client
        .post(&endpoint)
        .json(&json!({ "model": model }))
        .send()
        .await
        .with_context(|| format!("Ollama недоступен по {endpoint}"))?;
    decode(response, "/api/show").await
}

async fn decode(response: reqwest::Response, path: &str) -> Result<Value> {
    let status = response.status();
    let text = response.text().await.context("не прочитать ответ Ollama")?;
    if !status.is_success() {
        bail!("Ollama ответил {status} на {path}: {}", text.trim());
    }
    serde_json::from_str(&text).with_context(|| format!("неожиданный ответ {path}"))
}

/// Число из `model_info` по суффиксу ключа: префикс — архитектура модели
/// (`nomic-bert.embedding_length`, `bert.embedding_length`), заранее его не знаем.
fn info_number(show: &Value, suffix: &str) -> Option<u64> {
    show["model_info"]
        .as_object()?
        .iter()
        .find(|(key, _)| key.ends_with(suffix))
        .and_then(|(_, value)| value.as_u64())
}

/// Размерность вектора модели по `/api/show`; `None`, если метаданных нет.
pub async fn embedding_length(url: &str, model: &str) -> Result<Option<usize>> {
    let body = show(&client()?, url, model).await?;
    Ok(info_number(&body, ".embedding_length").map(|n| n as usize))
}

/// Модели Ollama с capability `embedding`. Capability есть только в
/// `/api/show`, поэтому по запросу на каждую модель из `/api/tags`; модель,
/// которую `show` не отдал, в список не попадает — подтвердить её нельзя.
/// Ollama старше версии с `capabilities` не покажет ни одной модели.
pub async fn embedding_models(url: &str) -> Result<Vec<EmbeddingModel>> {
    let client = client()?;
    let tags = get_json(&client, url, "/api/tags").await?;
    let mut out = Vec::new();
    for entry in tags["models"].as_array().into_iter().flatten() {
        let Some(name) = entry["name"].as_str() else {
            continue;
        };
        let Ok(body) = show(&client, url, name).await else {
            continue;
        };
        let is_embedding = body["capabilities"]
            .as_array()
            .is_some_and(|caps| caps.iter().any(|c| c == "embedding"));
        if is_embedding {
            out.push(EmbeddingModel {
                name: name.to_string(),
                dim: info_number(&body, ".embedding_length").map(|n| n as usize),
                context_length: info_number(&body, ".context_length"),
                size: entry["size"].as_u64(),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Переформулирует запрос через чат-модель. Ответ намеренно ограничивается
/// первой строкой: служебные пояснения модели не должны попасть в поиск.
pub async fn rewrite_query(url: &str, model: &str, query: &str) -> Result<String> {
    let client = client()?;
    let endpoint = format!("{}/api/chat", url.trim_end_matches('/'));
    let response = client
        .post(&endpoint)
        .json(&json!({
            "model": model,
            "stream": false,
            "options": {"temperature": 0, "num_predict": 128},
            "messages": [
                {"role": "system", "content": "Переформулируй запрос для семантического поиска. Верни только одну короткую поисковую формулировку без пояснений."},
                {"role": "user", "content": query}
            ]
        }))
        .send()
        .await
        .with_context(|| format!("Ollama недоступен по {endpoint}"))?;
    let body = decode(response, "/api/chat").await?;
    let raw = body["message"]["content"]
        .as_str()
        .or_else(|| body["response"].as_str())
        .context("Ollama не вернул текст rewrite")?;
    let cleaned = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .trim_matches('\x60')
        .trim()
        .trim_start_matches("Запрос:")
        .trim();
    if cleaned.is_empty() {
        bail!("Ollama вернул пустой rewrite")
    }
    Ok(cleaned.chars().take(512).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::path;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn rewrite_keeps_one_clean_search_line() {
        let server = MockServer::start().await;
        Mock::given(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "message": {"content": "Запрос:  query about UDP\nПояснение: лишнее"}
            })))
            .mount(&server)
            .await;
        assert_eq!(
            rewrite_query(&server.uri(), "rewrite-model", "как работает UDP")
                .await
                .unwrap(),
            "query about UDP"
        );
    }

    #[tokio::test]
    async fn rewrite_errors_on_empty_or_http_failure() {
        let server = MockServer::start().await;
        Mock::given(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "message": {"content": "  \n"}
            })))
            .mount(&server)
            .await;
        assert!(
            rewrite_query(&server.uri(), "rewrite-model", "q")
                .await
                .is_err()
        );

        let failed = MockServer::start().await;
        Mock::given(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503).set_body_string("offline"))
            .mount(&failed)
            .await;
        assert!(
            rewrite_query(&failed.uri(), "rewrite-model", "q")
                .await
                .is_err()
        );
    }
}
