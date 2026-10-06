//! `index-mcp serve` — MCP-сервер на stdio поверх тех же функций, что и CLI.
//!
//! Инструменты (имена и поля JSON — общий контракт с `agentcli`,
//! `crates/cli/src/index.rs`):
//!
//! - `index_search` — ближайшие чанки к запросу (читает);
//! - `index_status` — что лежит в базе (читает);
//! - `index_models` — модели Ollama с эмбеддингами (читает);
//! - `index_build` — построить индекс из каталога `.docx` (пишет в базу);
//! - `index_compare` — сравнить стратегии, записать отчёт (читает базу,
//!   пишет отчёт).
//!
//! Сборка и сравнение — это `run_build` и `run_compare` из `main.rs`, без
//! копии логики: инструмент лишь собирает для них те же структуры, что clap
//! собирает из флагов. Что не передано, берёт умолчания `build`; модель,
//! адрес Ollama и параметры эмбеддинга по умолчанию — флаги `serve`, чтобы
//! построение и поиск шли одной моделью.

use crate::search::{self, QueryModel};
use crate::store::Store;
use crate::{BuildArgs, CompareArgs, EmbedArgs, Unit, defaults, ollama};
use anyhow::{Context, Result, ensure};
use clap::Args;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::schemars::JsonSchema;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

/// Флаги `serve`. Размерности нет: поиск берёт её из базы, сборка — из
/// Ollama, так что `--dim` только вводил бы в заблуждение.
#[derive(Debug, Args, Clone)]
pub struct ServeArgs {
    /// Файл базы SQLite.
    #[arg(long, default_value = "index.db")]
    db: PathBuf,
    /// Стратегия поиска, если `strategy` не передан; без неё при нескольких
    /// стратегиях в базе поиск требует явного выбора.
    #[arg(long)]
    strategy: Option<String>,
    /// Адрес Ollama.
    #[arg(long, default_value = defaults::OLLAMA_URL)]
    ollama_url: String,
    /// Модель эмбеддингов запроса; по умолчанию она же — модель `index_build`.
    #[arg(long, default_value = defaults::MODEL)]
    model: String,
    /// Префикс задачи перед текстом чанка (для `index_build`).
    #[arg(long, default_value = defaults::DOC_PREFIX)]
    doc_prefix: String,
    /// Префикс задачи перед запросом.
    #[arg(long, default_value = defaults::QUERY_PREFIX)]
    query_prefix: String,
    /// Текстов в одном запросе к /api/embed.
    #[arg(long, default_value_t = defaults::BATCH)]
    batch: usize,
    /// Контекст модели в токенах.
    #[arg(long, default_value_t = defaults::NUM_CTX)]
    num_ctx: u32,
    /// Модель для необязательной переформулировки запроса.
    #[arg(long)]
    rewrite_model: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
struct SearchArgs {
    /// Question or phrase to look for, in natural language.
    query: String,
    /// Chunking strategy to search: "fixed" or "structure". Optional when the
    /// database holds one strategy or the server has a default.
    #[serde(default)]
    strategy: Option<String>,
    /// How many chunks to return (default 5, at most 20).
    #[serde(default)]
    top_k: Option<usize>,
    /// Сколько кандидатов взять до фильтрации (умолчание 20, максимум 100).
    #[serde(default)]
    candidate_top_k: Option<usize>,
    /// Минимальный cosine score; отсутствие отключает фильтр.
    #[serde(default)]
    similarity_threshold: Option<f32>,
    /// Переформулировать запрос через Ollama перед эмбеддингом.
    #[serde(default)]
    rewrite: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "lowercase")]
enum UnitArg {
    Chars,
    Tokens,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
struct BuildParams {
    /// Directory with .docx files (walked recursively).
    input: String,
    /// "fixed", "structure" or "all" (default).
    #[serde(default)]
    strategy: Option<String>,
    /// Unit of the sizes below: "chars" (default) or "tokens" of the embedding model.
    #[serde(default)]
    unit: Option<UnitArg>,
    /// fixed: window size (default 1200).
    #[serde(default)]
    chunk_size: Option<usize>,
    /// fixed: overlap of neighbouring windows (default 200).
    #[serde(default)]
    overlap: Option<usize>,
    /// structure: ceiling of a chunk (default 1500).
    #[serde(default)]
    max_section: Option<usize>,
    /// structure: shorter pieces are merged with a neighbour (default 200).
    #[serde(default)]
    min_section: Option<usize>,
    /// Minimum corpus size in characters (default 50000).
    #[serde(default)]
    min_chars: Option<usize>,
    /// Ollama embedding model (default: the server's --model, nomic-embed-text).
    #[serde(default)]
    model: Option<String>,
    /// Model context in tokens (default 2048).
    #[serde(default)]
    num_ctx: Option<u32>,
    /// Texts per /api/embed request (default 32).
    #[serde(default)]
    batch: Option<usize>,
    /// Vector length. Default: taken from the model (/api/show, else the
    /// first vector), not a fixed 768.
    #[serde(default)]
    dim: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
struct CompareParams {
    /// JSON file with questions and expected sections.
    questions: String,
    /// Where to write the Markdown report. Overwritten if it exists.
    out: String,
    /// Сколько результатов оставить после фильтрации.
    #[serde(default)]
    top_k: Option<usize>,
    /// Сколько кандидатов брать до фильтрации.
    #[serde(default)]
    candidate_top_k: Option<usize>,
    /// Минимальный cosine score; отсутствие отключает фильтр.
    #[serde(default)]
    similarity_threshold: Option<f32>,
    /// Модель для query rewrite; без неё используется модель сервера.
    #[serde(default)]
    rewrite_model: Option<String>,
}

#[derive(Debug, Clone)]
struct IndexServer {
    args: ServeArgs,
    tool_router: ToolRouter<Self>,
}

fn ok_json(value: &impl Serialize) -> CallToolResult {
    match serde_json::to_value(value) {
        Ok(value) => CallToolResult::structured(value),
        Err(err) => fail(format!("не удалось сериализовать результат: {err}")),
    }
}

fn fail(err: impl std::fmt::Display) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(err.to_string())])
}

/// Результат или ошибка со всей цепочкой причин (`{:#}`).
fn respond(result: Result<Value>) -> CallToolResult {
    match result {
        Ok(value) => ok_json(&value),
        Err(err) => fail(format!("{err:#}")),
    }
}

impl ServeArgs {
    fn query_model(&self) -> QueryModel {
        QueryModel {
            url: self.ollama_url.clone(),
            model: self.model.clone(),
            prefix: self.query_prefix.clone(),
            num_ctx: self.num_ctx,
            batch: self.batch,
            rewrite_model: self.rewrite_model.clone(),
        }
    }

    fn embed_args(&self, dim: usize) -> EmbedArgs {
        EmbedArgs {
            ollama_url: self.ollama_url.clone(),
            model: self.model.clone(),
            doc_prefix: self.doc_prefix.clone(),
            query_prefix: self.query_prefix.clone(),
            batch: self.batch,
            num_ctx: self.num_ctx,
            dim,
        }
    }
}

/// Размерность новой модели: `/api/show`, иначе длина первого вектора.
/// Умолчание 768 годится только `nomic-embed-text`; другая модель с ним
/// упала бы на сверке размерности или, хуже, записала бы неверную.
async fn resolve_dim(embed: &EmbedArgs) -> Result<usize> {
    if let Some(dim) = ollama::embedding_length(&embed.ollama_url, &embed.model)
        .await
        .ok()
        .flatten()
    {
        return Ok(dim);
    }
    let embedder = embed.embedder()?;
    embedder
        .probe_dim()
        .await
        .with_context(|| format!("размерность модели {}", embed.model))
}

impl IndexServer {
    async fn build(&self, p: BuildParams) -> Result<Value> {
        let mut embed = self.args.embed_args(defaults::DIM);
        if let Some(model) = p.model {
            embed.model = model;
        }
        if let Some(n) = p.num_ctx {
            embed.num_ctx = n;
        }
        if let Some(n) = p.batch {
            embed.batch = n;
        }
        embed.dim = match p.dim {
            Some(dim) => dim,
            None => resolve_dim(&embed).await?,
        };
        let dim = embed.dim;
        let args = BuildArgs {
            input: PathBuf::from(p.input),
            db: self.args.db.clone(),
            strategy: p.strategy.unwrap_or_else(|| defaults::STRATEGY.to_string()),
            unit: match p.unit {
                Some(UnitArg::Tokens) => Unit::Tokens,
                _ => Unit::Chars,
            },
            chunk_size: p.chunk_size.unwrap_or(defaults::CHUNK_SIZE),
            overlap: p.overlap.unwrap_or(defaults::OVERLAP),
            max_section: p.max_section.unwrap_or(defaults::MAX_SECTION),
            min_section: p.min_section.unwrap_or(defaults::MIN_SECTION),
            min_chars: p.min_chars.unwrap_or(defaults::MIN_CHARS),
            embed,
        };
        let infos = crate::run_build(&args).await?;
        Ok(json!({
            "db": args.db,
            "model": args.embed.model,
            "dim": dim,
            "strategies": infos.iter().map(|i| json!({
                "strategy": i.strategy,
                "chunks": i.chunks,
                "files": i.files,
                "chars": i.chars,
                "embed_ms": i.embed_ms,
            })).collect::<Vec<_>>(),
        }))
    }

    async fn status(&self) -> Result<Value> {
        let db = &self.args.db;
        let mut out = json!({
            "db": db,
            "exists": db.is_file(),
            "search_model": self.args.model,
            "strategies": [],
        });
        // Store::open создал бы пустую базу: статус не должен ничего писать.
        if !db.is_file() {
            return Ok(out);
        }
        let store = Store::open(db).await?;
        let times = store.build_times().await?;
        out["strategies"] = store
            .builds()
            .await?
            .into_iter()
            .map(|b| {
                json!({
                    "built_at": times.get(&b.strategy),
                    "strategy": b.strategy,
                    "chunks": b.chunks,
                    "files": b.files,
                    "chars": b.chars,
                    "model": b.model,
                    "dim": b.dim,
                    "embed_ms": b.embed_ms,
                    "params": serde_json::from_str::<Value>(&b.params).unwrap_or(Value::Null),
                })
            })
            .collect();
        Ok(out)
    }

    async fn compare(&self, p: CompareParams) -> Result<Value> {
        let db = &self.args.db;
        ensure!(
            db.is_file(),
            "нет базы {}: сначала index_build",
            db.display()
        );
        // Размерность — из базы: run_compare сверяет её с флагами.
        let builds = Store::open(db).await?.builds().await?;
        let dim = builds
            .first()
            .context("в базе нет построенных стратегий")?
            .dim as usize;
        let args = CompareArgs {
            db: db.clone(),
            questions: PathBuf::from(p.questions),
            out: PathBuf::from(p.out),
            embed: self.args.embed_args(dim),
            top_k: p.top_k.unwrap_or(search::DEFAULT_TOP_K),
            candidate_top_k: p.candidate_top_k.unwrap_or(search::DEFAULT_CANDIDATE_TOP_K),
            similarity_threshold: p
                .similarity_threshold
                .or(Some(search::DEFAULT_COMPARE_THRESHOLD)),
            rewrite_model: p.rewrite_model.or_else(|| self.args.rewrite_model.clone()),
        };
        let report = crate::run_compare(&args).await?;
        Ok(json!({ "out": args.out, "report": report }))
    }
}

#[tool_router]
impl IndexServer {
    fn new(args: ServeArgs) -> Self {
        Self {
            args,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Semantic search over the document index. Optional candidate_top_k limits candidates before \
similarity_threshold filtering; top_k limits final hits; rewrite asks Ollama for a search formulation. Returns \
{query,used_query,strategy,model,dim,top_k,candidate_top_k,similarity_threshold,rewrite,rewrite_fallback,candidates,results,\
hits:[{chunk_id,source,section,score,text}]}. Fails if the query model differs from the built model."
    )]
    async fn index_search(&self, Parameters(p): Parameters<SearchArgs>) -> CallToolResult {
        let options = search::SearchOptions {
            top_k: p.top_k.unwrap_or(search::DEFAULT_TOP_K),
            candidate_top_k: p.candidate_top_k.unwrap_or(search::DEFAULT_CANDIDATE_TOP_K),
            similarity_threshold: p.similarity_threshold,
        };
        let strategy = p.strategy.or_else(|| self.args.strategy.clone());
        respond(
            search::search(
                &self.args.db,
                &self.args.query_model(),
                &p.query,
                strategy.as_deref(),
                options,
                p.rewrite,
            )
            .await
            .and_then(|out| Ok(serde_json::to_value(out)?)),
        )
    }

    #[tool(
        description = "What the index holds. Returns {db, exists, search_model, strategies:[{strategy, chunks, \
files, chars, model, dim, embed_ms, built_at, params}]}."
    )]
    async fn index_status(&self) -> CallToolResult {
        respond(self.status().await)
    }

    #[tool(
        description = "Ollama models that can produce embeddings. Returns {models:[{name, dim, context_length, size}]}."
    )]
    async fn index_models(&self) -> CallToolResult {
        respond(
            ollama::embedding_models(&self.args.ollama_url)
                .await
                .map(|models| json!({ "models": models })),
        )
    }

    #[tool(
        description = "WRITES the index: reads .docx files from `input`, chunks them, embeds the chunks with \
Ollama and replaces the chosen strategies in the database. Unset parameters take the defaults of `index-mcp build`; \
the vector length comes from the model. Slow: minutes on a large corpus. Returns \
{db, model, dim, strategies:[{strategy, chunks, files, chars, embed_ms}]}."
    )]
    async fn index_build(&self, Parameters(p): Parameters<BuildParams>) -> CallToolResult {
        respond(self.build(p).await)
    }

    #[tool(
        description = "Compares baseline, rewrite, filter and rewrite+filter on one questions file and WRITES a \
Markdown report to out. Optional top_k, candidate_top_k, similarity_threshold and rewrite_model set the shared \
comparison parameters. Returns {out, report}."
    )]
    async fn index_compare(&self, Parameters(p): Parameters<CompareParams>) -> CallToolResult {
        respond(self.compare(p).await)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for IndexServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(format!(
                "Document index in {}: index_search finds chunks by meaning, index_status shows what is built, \
index_models lists embedding models, index_build (re)builds the index from a directory of .docx.",
                self.args.db.display()
            ))
    }
}

pub async fn run(args: ServeArgs) -> Result<()> {
    // Рукопожатие первым: stdout принадлежит протоколу, всё остальное —
    // в stderr (прогресс сборки).
    let service = IndexServer::new(args)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}
