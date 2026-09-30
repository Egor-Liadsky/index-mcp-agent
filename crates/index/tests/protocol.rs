//! Сервер проверяется так, как его видит MCP-клиент: настоящий процесс
//! `index-mcp serve`, клиент `rmcp` через stdin/stdout, Ollama подменён
//! `wiremock`, `.docx` собираются в памяти.
//!
//! Поддельный эмбеддер — «мешок слов»: вопросы находят раздел с теми же
//! словами. Размерность зависит от модели, чтобы проверять и сверку моделей,
//! и определение размерности новой модели.

use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const BINARY: &str = env!("CARGO_BIN_EXE_index-mcp");

/// Длина вектора по модели: у `nomic-embed-text` — 768, как у умолчания
/// `build`, у остальных — иная, чтобы умолчание 768 не могло сойти за верное.
fn model_dim(model: &str) -> usize {
    match model {
        "nomic-embed-text" => 768,
        "mini-embed" => 32,
        _ => 48, // noshow-embed: /api/show о нём молчит
    }
}

fn vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 2)
    {
        let h = word
            .to_lowercase()
            .bytes()
            .fold(1469598103934665603u64, |h, b| {
                (h ^ b as u64).wrapping_mul(1099511628211)
            });
        v[(h % dim as u64) as usize] += 1.0;
    }
    v[0] += 0.01;
    v
}

struct FakeEmbed;

impl Respond for FakeEmbed {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let dim = model_dim(body["model"].as_str().unwrap());
        let embeddings: Vec<Vec<f32>> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| vector(t.as_str().unwrap(), dim))
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({ "embeddings": embeddings }))
    }
}

struct FakeShow;

impl Respond for FakeShow {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let (caps, info) = match body["model"].as_str().unwrap() {
            "nomic-embed-text:latest" | "nomic-embed-text" => (
                json!(["embedding"]),
                json!({ "nomic-bert.embedding_length": 768, "nomic-bert.context_length": 2048 }),
            ),
            "mini-embed" => (json!(["embedding"]), json!({ "bert.embedding_length": 32 })),
            "llama3" => (
                json!(["completion"]),
                json!({ "llama.embedding_length": 4096 }),
            ),
            _ => return ResponseTemplate::new(404).set_body_json(json!({ "error": "not found" })),
        };
        ResponseTemplate::new(200)
            .set_body_json(json!({ "capabilities": caps, "model_info": info }))
    }
}

async fn ollama() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(path("/api/embed"))
        .respond_with(FakeEmbed)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(FakeShow)
        .mount(&server)
        .await;
    Mock::given(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "models": [
            { "name": "nomic-embed-text:latest", "size": 274_302_450u64 },
            { "name": "llama3", "size": 4_000_000_000u64 },
            { "name": "mini-embed", "size": 1000 },
            { "name": "vanished", "size": 1 },
        ]})))
        .mount(&server)
        .await;
    server
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("index-mcp-{name}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn heading(text: &str) -> String {
    format!(r#"<w:p><w:pPr><w:pStyle w:val="H1"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#)
}

fn para(text: &str, times: usize) -> String {
    let text = vec![text; times].join(" ");
    format!(r#"<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>"#)
}

fn write_docx(path: &Path, body: &str) {
    const NS: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main""#;
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default();
    zip.start_file("word/document.xml", opts).unwrap();
    write!(
        zip,
        r#"<?xml version="1.0"?><w:document {NS}><w:body>{body}<w:sectPr/></w:body></w:document>"#
    )
    .unwrap();
    zip.start_file("word/styles.xml", opts).unwrap();
    write!(
        zip,
        r#"<?xml version="1.0"?><w:styles {NS}><w:style w:type="paragraph" w:styleId="H1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>"#
    )
    .unwrap();
    zip.finish().unwrap();
}

/// Каталог с двумя конспектами; вопросы про UDP и про память находят свои разделы.
fn corpus() -> PathBuf {
    let dir = temp_dir("corpus");
    let net = [
        heading("TCP"),
        para(
            "Протокол TCP устанавливает соединение и гарантирует доставку байтов по порядку.",
            20,
        ),
        heading("UDP"),
        para(
            "Протокол UDP отправляет датаграммы без соединения и без подтверждений.",
            20,
        ),
    ]
    .concat();
    write_docx(&dir.join("сети.docx"), &net);
    let os = [
        heading("Процессы"),
        para(
            "Планировщик выбирает процесс для исполнения на процессоре по приоритету.",
            20,
        ),
        heading("Память"),
        para(
            "Виртуальная память отображает страницы на физические кадры через таблицу страниц.",
            20,
        ),
    ]
    .concat();
    write_docx(&dir.join("ос.docx"), &os);
    dir
}

type Client = RunningService<RoleClient, ()>;

async fn connect(db: &Path, ollama_url: &str, extra: &[&str]) -> Client {
    let mut command = tokio::process::Command::new(BINARY);
    command
        .arg("serve")
        .arg("--db")
        .arg(db)
        .arg("--ollama-url")
        .arg(ollama_url)
        .args(extra);
    let (transport, _stderr) = TokioChildProcess::builder(command)
        .stderr(Stdio::null())
        .spawn()
        .expect("запуск сервера");
    ().serve(transport).await.expect("рукопожатие MCP")
}

/// Структурированный результат или текст ошибки.
async fn call(client: &Client, name: &str, arguments: Value) -> Result<Value, String> {
    let params = CallToolRequestParams::new(name.to_string())
        .with_arguments(arguments.as_object().cloned().unwrap_or_default());
    let result = client.peer().call_tool(params).await.expect("tools/call");
    let content = serde_json::to_value(&result.content).unwrap();
    let text = content[0]["text"].as_str().unwrap_or_default().to_string();
    if result.is_error == Some(true) {
        return Err(text);
    }
    let structured = result.structured_content.expect("structuredContent");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), structured);
    Ok(structured)
}

/// Параметры сборки, которым хватает маленького корпуса теста.
fn small_build(input: &Path, extra: Value) -> Value {
    let mut args = json!({
        "input": input,
        "min_chars": 0,
        "chunk_size": 600,
        "overlap": 100,
        "max_section": 1500,
    });
    args.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    args
}

#[tokio::test]
async fn lists_the_index_tools() {
    let server = ollama().await;
    let client = connect(&temp_dir("list").join("i.db"), &server.uri(), &[]).await;
    let mut names: Vec<String> = client
        .peer()
        .list_all_tools()
        .await
        .unwrap()
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "index_build",
            "index_compare",
            "index_models",
            "index_search",
            "index_status"
        ]
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn build_status_search_compare_roundtrip() {
    let server = ollama().await;
    let db = temp_dir("flow").join("i.db");
    let client = connect(&db, &server.uri(), &[]).await;

    // Статус пустой базы ничего не создаёт.
    let status = call(&client, "index_status", json!({})).await.unwrap();
    assert_eq!(status["exists"], false);
    assert!(!db.exists());
    let err = call(&client, "index_search", json!({ "query": "UDP" }))
        .await
        .unwrap_err();
    assert!(err.contains("нет базы"), "{err}");
    assert!(!db.exists());

    // Умолчания build: модель nomic-embed-text, dim берётся у модели.
    let built = call(&client, "index_build", small_build(&corpus(), json!({})))
        .await
        .unwrap();
    assert_eq!(built["model"], "nomic-embed-text");
    assert_eq!(built["dim"], 768);
    let strategies: Vec<&str> = built["strategies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["strategy"].as_str().unwrap())
        .collect();
    assert_eq!(strategies, ["fixed", "structure"]);

    let status = call(&client, "index_status", json!({})).await.unwrap();
    assert_eq!(status["exists"], true);
    let structure = &status["strategies"][1];
    assert_eq!(structure["strategy"], "structure");
    assert_eq!(structure["model"], "nomic-embed-text");
    assert_eq!(structure["dim"], 768);
    assert_eq!(structure["files"], 2);
    assert_eq!(structure["chunks"], built["strategies"][1]["chunks"]);
    assert_eq!(structure["params"]["num_ctx"], 2048);
    assert!(structure["built_at"].as_str().unwrap().starts_with("20"));

    // Стратегий две, и не указанная — ошибка, а не молчаливый выбор.
    let err = call(&client, "index_search", json!({ "query": "UDP" }))
        .await
        .unwrap_err();
    assert!(err.contains("укажи strategy"), "{err}");

    let found = call(
        &client,
        "index_search",
        json!({ "query": "Как UDP отправляет датаграммы без подтверждений?",
                "strategy": "structure", "top_k": 2 }),
    )
    .await
    .unwrap();
    assert_eq!(found["strategy"], "structure");
    assert_eq!(found["model"], "nomic-embed-text");
    let hits = found["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["section"], "UDP");
    assert!(
        hits[0]["chunk_id"]
            .as_str()
            .unwrap()
            .starts_with("structure:сети:")
    );
    assert!(hits[0]["source"].as_str().unwrap().ends_with("сети.docx"));
    assert!(hits[0]["text"].as_str().unwrap().contains("датаграммы"));
    assert!(hits[0]["score"].as_f64().unwrap() > hits[1]["score"].as_f64().unwrap());

    // Отчёт пишется туда, куда сказано, и повторяет то, что вернул инструмент.
    let dir = temp_dir("questions");
    let questions = dir.join("q.json");
    std::fs::write(
        &questions,
        r#"[{"question": "Как UDP отправляет датаграммы без подтверждений?", "expected_section": "UDP"},
            {"question": "Что делает планировщик процессов?", "expected_section": "Процессы"}]"#,
    )
    .unwrap();
    let out = dir.join("report.md");
    let compared = call(
        &client,
        "index_compare",
        json!({ "questions": questions, "out": out }),
    )
    .await
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(&out).unwrap(),
        compared["report"].as_str().unwrap()
    );
    assert!(compared["report"].as_str().unwrap().contains("MRR"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn search_refuses_a_different_model() {
    let server = ollama().await;
    let db = temp_dir("mismatch").join("i.db");
    // Индекс строится «мелкой» моделью с размерностью 32 из /api/show…
    let builder = connect(&db, &server.uri(), &[]).await;
    let built = call(
        &builder,
        "index_build",
        small_build(
            &corpus(),
            json!({ "model": "mini-embed", "strategy": "structure" }),
        ),
    )
    .await
    .unwrap();
    assert_eq!(built["model"], "mini-embed");
    assert_eq!(
        built["dim"], 32,
        "размерность — у модели, а не умолчание 768"
    );
    builder.cancel().await.unwrap();

    // …а сервер по умолчанию ищет nomic-embed-text: результата нет, есть ошибка.
    let client = connect(&db, &server.uri(), &[]).await;
    let err = call(&client, "index_search", json!({ "query": "UDP" }))
        .await
        .unwrap_err();
    assert!(err.contains("построена моделью mini-embed"), "{err}");
    assert!(err.contains("векторы несравнимы"), "{err}");
    client.cancel().await.unwrap();

    // С той же моделью поиск работает; единственная стратегия выбирается сама.
    let client = connect(&db, &server.uri(), &["--model", "mini-embed"]).await;
    let found = call(
        &client,
        "index_search",
        json!({ "query": "Виртуальная память" }),
    )
    .await
    .unwrap();
    assert_eq!(found["dim"], 32);
    assert_eq!(found["hits"][0]["section"], "Память");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn dim_falls_back_to_the_first_vector() {
    let server = ollama().await;
    let db = temp_dir("probe").join("i.db");
    let client = connect(&db, &server.uri(), &[]).await;
    // /api/show для этой модели даёт 404: размерность берётся из первого вектора.
    let built = call(
        &client,
        "index_build",
        small_build(
            &corpus(),
            json!({ "model": "noshow-embed", "strategy": "structure" }),
        ),
    )
    .await
    .unwrap();
    assert_eq!(built["dim"], 48);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn build_uses_build_defaults() {
    let server = ollama().await;
    let db = temp_dir("defaults").join("i.db");
    let client = connect(&db, &server.uri(), &[]).await;
    // min_chars не передан: действует умолчание build в 50 000 символов.
    let err = call(&client, "index_build", json!({ "input": corpus() }))
        .await
        .unwrap_err();
    assert!(err.contains("корпус слишком мал"), "{err}");
    assert!(err.contains("нужно не меньше 50000"), "{err}");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn lists_only_embedding_models() {
    let server = ollama().await;
    let client = connect(&temp_dir("models").join("i.db"), &server.uri(), &[]).await;
    let models = call(&client, "index_models", json!({})).await.unwrap();
    assert_eq!(
        models["models"],
        json!([
            { "name": "mini-embed", "dim": 32, "context_length": null, "size": 1000 },
            { "name": "nomic-embed-text:latest", "dim": 768, "context_length": 2048,
              "size": 274_302_450u64 },
        ])
    );
    client.cancel().await.unwrap();
}
