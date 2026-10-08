//! `index-mcp` — локальный индекс конспектов `.docx` для поиска по смыслу.
//!
//! Две команды:
//!
//! - `build` — читает `.docx` из каталога, режет их стратегиями chunking
//!   (трейт [`chunk::Chunker`]), получает эмбеддинги у локального Ollama и
//!   пишет чанки с векторами в SQLite;
//! - `compare` — сравнивает построенные стратегии по форме чанков и по
//!   качеству поиска на вопросах из `questions.json` и пишет отчёт в
//!   Markdown.
//!
//! Индексатор ниже знает только трейт `Chunker`, [`embed::Embedder`] и
//! [`store::Store`]: новая стратегия не меняет ни его, ни эмбеддер, ни
//! хранилище. Инструмент MCP для поиска по индексу появится отдельно;
//! имя бинарника уже отражает, что это сервер из каталога `mcp/`.

mod chunk;
mod compare;
mod docx;
mod embed;
mod ollama;
mod search;
mod serve;
mod store;
mod tokens;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chunk::units::{Chars, Units};
use chunk::{ChunkParams, Chunker};
use docx::Document;
use embed::{ContextOverflow, EmbedConfig, Embedder};
use store::{BuildInfo, ChunkRow, Store};

/// Умолчания `build` в одном месте: их берут и флаги CLI, и инструмент
/// `index_build`, у которого не переданное значение — это умолчание команды.
mod defaults {
    pub const OLLAMA_URL: &str = "http://localhost:11434";
    pub const MODEL: &str = "nomic-embed-text";
    pub const DOC_PREFIX: &str = "search_document: ";
    pub const QUERY_PREFIX: &str = "search_query: ";
    pub const BATCH: usize = 32;
    pub const NUM_CTX: u32 = 2048;
    pub const DIM: usize = 768;
    pub const STRATEGY: &str = "all";
    pub const CHUNK_SIZE: usize = 1200;
    pub const OVERLAP: usize = 200;
    pub const MAX_SECTION: usize = 1500;
    pub const MIN_SECTION: usize = 200;
    pub const MIN_CHARS: usize = 50_000;
}

#[derive(Debug, Parser)]
#[command(
    name = "index-mcp",
    version,
    about = "Локальный индекс .docx по стратегиям chunking"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Построить индекс: .docx → чанки → эмбеддинги → SQLite.
    Build(BuildArgs),
    /// Сравнить стратегии по базе и вопросам, записать отчёт.
    Compare(CompareArgs),
    /// MCP-сервер на stdio: поиск по индексу, статус, модели, сборка.
    Serve(serve::ServeArgs),
}

#[derive(Debug, Args)]
struct EmbedArgs {
    /// Адрес Ollama.
    #[arg(long, default_value = defaults::OLLAMA_URL)]
    ollama_url: String,
    /// Модель эмбеддингов Ollama.
    #[arg(long, default_value = defaults::MODEL)]
    model: String,
    /// Префикс задачи перед текстом чанка (в базу не пишется).
    #[arg(long, default_value = defaults::DOC_PREFIX)]
    doc_prefix: String,
    /// Префикс задачи перед вопросом.
    #[arg(long, default_value = defaults::QUERY_PREFIX)]
    query_prefix: String,
    /// Текстов в одном запросе к /api/embed.
    #[arg(long, default_value_t = defaults::BATCH)]
    batch: usize,
    /// Контекст модели в токенах (options.num_ctx); больше обученного
    /// контекста модели (2048 у nomic-embed-text) Ollama его не расширяет.
    #[arg(long, default_value_t = defaults::NUM_CTX)]
    num_ctx: u32,
    /// Ожидаемая размерность вектора.
    #[arg(long, default_value_t = defaults::DIM)]
    dim: usize,
}

impl EmbedArgs {
    fn embedder(&self) -> Result<Embedder> {
        Embedder::new(EmbedConfig {
            url: self.ollama_url.clone(),
            model: self.model.clone(),
            batch: self.batch,
            num_ctx: self.num_ctx,
            dim: self.dim,
        })
    }
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Каталог с .docx (обходится рекурсивно).
    #[arg(long)]
    input: PathBuf,
    /// Файл базы SQLite.
    #[arg(long, default_value = "index.db")]
    db: PathBuf,
    /// Стратегия: fixed, structure или all.
    #[arg(long, default_value = defaults::STRATEGY)]
    strategy: String,
    /// В чём меряются размеры ниже: символы или токены модели.
    #[arg(long, value_enum, default_value_t = Unit::Chars)]
    unit: Unit,
    /// fixed: размер окна в единицах --unit.
    #[arg(long, default_value_t = defaults::CHUNK_SIZE)]
    chunk_size: usize,
    /// fixed: перекрытие соседних окон в единицах --unit.
    #[arg(long, default_value_t = defaults::OVERLAP)]
    overlap: usize,
    /// structure: потолок чанка в единицах --unit; раздел длиннее режется по
    /// абзацам. Умолчание в символах подобрано под контекст nomic-embed-text
    /// в 2048 токенов на русском тексте.
    #[arg(long, default_value_t = defaults::MAX_SECTION)]
    max_section: usize,
    /// structure: кусок короче (в единицах --unit) склеивается с соседним.
    #[arg(long, default_value_t = defaults::MIN_SECTION)]
    min_section: usize,
    /// Минимальный объём корпуса в символах.
    #[arg(long, default_value_t = defaults::MIN_CHARS)]
    min_chars: usize,
    #[command(flatten)]
    embed: EmbedArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
enum Unit {
    /// Символы текста.
    Chars,
    /// Токены модели эмбеддингов (словарь берётся у Ollama).
    Tokens,
}

#[derive(Debug, Args)]
struct CompareArgs {
    /// Файл базы SQLite, построенной командой build.
    #[arg(long, default_value = "index.db")]
    db: PathBuf,
    /// Вопросы с ожидаемыми разделами.
    #[arg(long, default_value = "questions.json")]
    questions: PathBuf,
    /// Куда записать отчёт.
    #[arg(long, default_value = "docs/chunking-comparison.md")]
    out: PathBuf,
    #[command(flatten)]
    embed: EmbedArgs,
    /// Сколько результатов оставить после фильтрации.
    #[arg(long, default_value_t = search::DEFAULT_TOP_K)]
    top_k: usize,
    /// Сколько кандидатов брать до фильтрации.
    #[arg(long, default_value_t = search::DEFAULT_CANDIDATE_TOP_K)]
    candidate_top_k: usize,
    /// Минимальный cosine score; отсутствие отключает фильтр.
    #[arg(long)]
    similarity_threshold: Option<f32>,
    /// Модель для query rewrite; по умолчанию модель эмбеддингов.
    #[arg(long)]
    rewrite_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build(args) => {
            for info in run_build(&args).await? {
                println!(
                    "{}: {} чанков из {} файлов ({} символов), эмбеддинг {} мс",
                    info.strategy, info.chunks, info.files, info.chars, info.embed_ms
                );
            }
        }
        Command::Compare(args) => {
            let report = run_compare(&args).await?;
            println!("{report}");
            println!("Отчёт записан в {}", args.out.display());
        }
        Command::Serve(args) => serve::run(args).await?,
    }
    Ok(())
}

/// `.docx` каталога рекурсивно, в детерминированном порядке. Пропускаются
/// скрытые каталоги и `~$*.docx` — lock-файлы открытого в Word документа.
fn collect_docx(dir: &Path) -> Result<Vec<PathBuf>> {
    ensure!(dir.is_dir(), "{} — не каталог", dir.display());
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)
            .with_context(|| format!("не прочитать {}", current.display()))?
        {
            let path = entry?.path();
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if name.to_lowercase().ends_with(".docx") && !name.starts_with("~$") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn read_corpus(args: &BuildArgs) -> Result<Vec<Document>> {
    let files = collect_docx(&args.input)?;
    ensure!(
        !files.is_empty(),
        "в {} нет файлов .docx",
        args.input.display()
    );
    let docs = files
        .iter()
        .map(|f| docx::read_docx(f))
        .collect::<Result<Vec<_>>>()?;
    // chunk_id строится из имени файла без расширения; одинаковые имена в
    // разных подкаталогах дали бы одинаковые id.
    let mut stems: HashMap<String, &Path> = HashMap::new();
    for d in &docs {
        if let Some(other) = stems.insert(d.file_stem(), &d.source) {
            bail!(
                "одинаковое имя файла {:?} у {} и {}: chunk_id не будет уникальным",
                d.file_stem(),
                other.display(),
                d.source.display()
            );
        }
    }
    let total: usize = docs.iter().map(Document::char_len).sum();
    ensure!(
        total >= args.min_chars,
        "корпус слишком мал: {total} символов в {} файлах, нужно не меньше {}",
        docs.len(),
        args.min_chars
    );
    Ok(docs)
}

/// Чанки стратегии по всем документам с `chunk_id` и метриками границ.
fn chunk_rows(chunker: &dyn Chunker, docs: &[Document]) -> Vec<ChunkRow> {
    let mut rows = Vec::new();
    for doc in docs {
        let stem = doc.file_stem();
        for (ordinal, c) in chunker.chunk(doc).into_iter().enumerate() {
            rows.push(ChunkRow {
                chunk_id: format!("{}:{stem}:{ordinal:04}", chunker.name()),
                strategy: chunker.name().to_string(),
                source: doc.source.display().to_string(),
                title: doc.title.clone(),
                crosses_section: chunk::crosses_section(doc, &c),
                cut_mid_sentence: chunk::cut_mid_sentence(doc, &c),
                section: c.section,
                ordinal: ordinal as i64,
                char_start: c.char_start as i64,
                char_end: c.char_end as i64,
                text: c.text,
            });
        }
    }
    rows
}

/// Единицы размеров. В токенах заодно проверяется, что самый большой чанк
/// вместе с префиксом задачи и служебными `[CLS]`/`[SEP]` помещается в
/// `--num-ctx`: иначе ошибка всплыла бы только на эмбеддинге.
async fn make_units(args: &BuildArgs) -> Result<Arc<dyn Units>> {
    match args.unit {
        Unit::Chars => Ok(Arc::new(Chars)),
        Unit::Tokens => {
            let wp = tokens::WordPiece::from_ollama(&args.embed.ollama_url, &args.embed.model)
                .await
                .context("словарь для --unit tokens")?;
            let overhead = wp.count(&args.embed.doc_prefix) + 2;
            let largest = args.chunk_size.max(args.max_section);
            ensure!(
                largest + overhead <= args.embed.num_ctx as usize,
                "чанк до {largest} токенов плюс {overhead} на префикс и служебные токены не помещается в --num-ctx {}",
                args.embed.num_ctx
            );
            Ok(Arc::new(tokens::TokenUnits(wp)))
        }
    }
}

async fn run_build(args: &BuildArgs) -> Result<Vec<BuildInfo>> {
    let embedder = args.embed.embedder()?;
    let units = make_units(args).await?;
    let params = ChunkParams {
        chunk_size: args.chunk_size,
        overlap: args.overlap,
        max_section: args.max_section,
        min_section: args.min_section,
        units,
    };
    let chunkers = chunk::select(&args.strategy, &params)?;
    let docs = read_corpus(args)?;
    let chars: usize = docs.iter().map(Document::char_len).sum();
    // Ход сборки — в stderr: stdout у `serve` занят протоколом, а у `build`
    // там итог.
    eprintln!("прочитано {} файлов, {chars} символов", docs.len());
    let store = Store::open(&args.db).await?;
    let mut infos = Vec::new();
    for chunker in &chunkers {
        let rows = chunk_rows(chunker.as_ref(), &docs);
        let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
        eprintln!(
            "{}: {} чанков, эмбеддинг моделью {}…",
            chunker.name(),
            rows.len(),
            embedder.model()
        );
        let started = Instant::now();
        // Эмбеддинг — до транзакции: недоступный Ollama не трогает прежний индекс.
        let vectors = embedder
            .embed(&args.embed.doc_prefix, &texts)
            .await
            .map_err(|err| {
                let hint = err.downcast_ref::<ContextOverflow>().map(|o| {
                    format!(
                        "чанк {} ({} символов) не помещается в контекст модели: уменьшите --max-section или --chunk-size",
                        rows[o.index].chunk_id,
                        rows[o.index].text.chars().count()
                    )
                });
                match hint {
                    Some(hint) => err.context(hint),
                    None => err,
                }
            })
            .with_context(|| format!("эмбеддинг стратегии {}", chunker.name()))?;
        let embed_ms = started.elapsed().as_millis() as i64;
        let info = BuildInfo {
            strategy: chunker.name().to_string(),
            model: embedder.model().to_string(),
            dim: embedder.dim() as i64,
            files: docs.len() as i64,
            chars: chars as i64,
            chunks: rows.len() as i64,
            embed_ms,
            params: serde_json::json!({
                "chunker": chunker.params(),
                "doc_prefix": args.embed.doc_prefix,
                "num_ctx": args.embed.num_ctx,
            })
            .to_string(),
        };
        store.replace_strategy(&info, &rows, &vectors).await?;
        infos.push(info);
    }
    Ok(infos)
}

async fn run_compare(args: &CompareArgs) -> Result<String> {
    let compare_threshold = args
        .similarity_threshold
        .unwrap_or(search::DEFAULT_COMPARE_THRESHOLD);
    search::validate_options(search::SearchOptions {
        top_k: args.top_k,
        candidate_top_k: args.candidate_top_k,
        similarity_threshold: Some(compare_threshold),
        rerank: true,
    })?;
    // Без этой проверки Store::open создал бы пустую базу и отчёт был бы пустым.
    ensure!(
        args.db.is_file(),
        "нет базы {}: сначала index-mcp build",
        args.db.display()
    );
    let store = Store::open(&args.db).await?;
    let builds = store.builds().await?;
    ensure!(
        !builds.is_empty(),
        "в {} нет построенных стратегий",
        args.db.display()
    );
    for b in &builds {
        ensure!(
            b.model == args.embed.model && b.dim as usize == args.embed.dim,
            "стратегия {} построена моделью {} (dim {}), а вопросы эмбеддятся моделью {} (dim {}): векторы несравнимы",
            b.strategy,
            b.model,
            b.dim,
            args.embed.model,
            args.embed.dim
        );
    }
    let questions = compare::load_questions(&args.questions)?;
    let embedder = args.embed.embedder()?;
    let texts: Vec<&str> = questions.iter().map(|q| q.question.as_str()).collect();
    let query_vectors = embedder
        .embed(&args.embed.query_prefix, &texts)
        .await
        .context("эмбеддинг вопросов")?;
    let rewrite_model = args.rewrite_model.as_deref().unwrap_or(&args.embed.model);
    let mut rewritten = Vec::with_capacity(questions.len());
    let mut rewrite_fallbacks = 0;
    for question in &questions {
        match ollama::rewrite_query(&args.embed.ollama_url, rewrite_model, &question.question).await
        {
            Ok(query) => rewritten.push(query),
            Err(_) => {
                rewrite_fallbacks += 1;
                rewritten.push(question.question.clone());
            }
        }
    }
    let rewritten_texts: Vec<&str> = rewritten.iter().map(String::as_str).collect();
    let rewritten_vectors = embedder
        .embed(&args.embed.query_prefix, &rewritten_texts)
        .await
        .context("эмбеддинг rewritten вопросов")?;
    let mut evals = Vec::new();
    let mut mode_evals = Vec::new();
    for b in builds {
        let chunks = store.load(&b.strategy).await?;
        // builds хранит последнее построение, а модель записана и при каждом векторе:
        // сверка по векторам ловит базу, собранную из частей разными моделями.
        if let Some(bad) = chunks
            .iter()
            .find(|c| c.model != args.embed.model || c.vector.len() != args.embed.dim)
        {
            bail!(
                "вектор чанка {} построен моделью {} (dim {}), а не {} (dim {})",
                bad.row.chunk_id,
                bad.model,
                bad.vector.len(),
                args.embed.model,
                args.embed.dim
            );
        }
        let strategy = b.strategy.clone();
        evals.push(compare::evaluate(
            &strategy,
            &chunks,
            Some(b),
            &questions,
            &query_vectors,
        ));
        let original_queries: Vec<String> = questions.iter().map(|q| q.question.clone()).collect();
        let modes = [
            (
                compare::ModeParams {
                    rewrite: false,
                    candidate_top_k: args.candidate_top_k,
                    top_k: args.top_k,
                    similarity_threshold: None,
                },
                &query_vectors,
                0,
            ),
            (
                compare::ModeParams {
                    rewrite: true,
                    candidate_top_k: args.candidate_top_k,
                    top_k: args.top_k,
                    similarity_threshold: None,
                },
                &rewritten_vectors,
                rewrite_fallbacks,
            ),
            (
                compare::ModeParams {
                    rewrite: false,
                    candidate_top_k: args.candidate_top_k,
                    top_k: args.top_k,
                    similarity_threshold: Some(compare_threshold),
                },
                &query_vectors,
                0,
            ),
            (
                compare::ModeParams {
                    rewrite: true,
                    candidate_top_k: args.candidate_top_k,
                    top_k: args.top_k,
                    similarity_threshold: Some(compare_threshold),
                },
                &rewritten_vectors,
                rewrite_fallbacks,
            ),
        ];
        for (params, vectors, fallbacks) in modes {
            let used_queries = if params.rewrite {
                &rewritten
            } else {
                &original_queries
            };
            mode_evals.push(compare::evaluate_mode(
                &strategy,
                &chunks,
                &questions,
                vectors,
                used_queries,
                params,
                fallbacks,
            ));
        }
    }
    let mut report = compare::render(&evals, &questions, &args.embed.model);
    report.push_str(&compare::render_modes(&mode_evals, &questions));
    report.push_str(&format!(
        "\nПараметры режимов: candidate_top_k={}, top_k={}, similarity_threshold={}, rewrite_model={}, rewrite_fallbacks={}.\n",
        args.candidate_top_k,
        args.top_k,
        compare_threshold,
        rewrite_model,
        rewrite_fallbacks,
    ));
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("не создать {}", parent.display()))?;
    }
    std::fs::write(&args.out, &report)
        .with_context(|| format!("не записать {}", args.out.display()))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use docx::testutil::{heading, para, russian_styles, write_docx};
    use embed::testutil::BagOfWords;
    use wiremock::matchers::path;
    use wiremock::{Mock, MockServer};

    fn build_args(extra: &[&str]) -> BuildArgs {
        let mut argv = vec!["index-mcp", "build"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Build(a) => a,
            _ => unreachable!(),
        }
    }

    fn compare_args(extra: &[&str]) -> CompareArgs {
        let mut argv = vec!["index-mcp", "compare"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Compare(a) => a,
            _ => unreachable!(),
        }
    }

    fn repeat(sentence: &str, n: usize) -> String {
        vec![sentence; n].join(" ")
    }

    /// Два конспекта с разделами на разные темы: у поддельного эмбеддера
    /// («мешок слов») вопросы находят свой раздел.
    fn write_corpus(dir: &Path) {
        let net = [
            heading(1, "Сети"),
            para(&repeat("Сеть связывает компьютеры каналами связи.", 3)),
            heading(2, "TCP"),
            para(&repeat(
                "Протокол TCP устанавливает соединение и гарантирует доставку байтов по порядку.",
                20,
            )),
            para(&repeat(
                "Окно перегрузки TCP растёт при подтверждениях и сжимается при потерях.",
                20,
            )),
            heading(2, "UDP"),
            para(&repeat(
                "Протокол UDP отправляет датаграммы без соединения и без подтверждений.",
                25,
            )),
        ]
        .concat();
        write_docx(
            &dir.join("сети.docx"),
            &russian_styles(),
            &net,
            Some("Компьютерные сети"),
        );
        let os = [
            heading(1, "Процессы"),
            para(&repeat(
                "Планировщик выбирает процесс для исполнения на процессоре по приоритету.",
                30,
            )),
            heading(1, "Память"),
            para(&repeat(
                "Виртуальная память отображает страницы на физические кадры через таблицу страниц.",
                30,
            )),
        ]
        .concat();
        write_docx(&dir.join("ос.docx"), &russian_styles(), &os, None);
    }

    #[test]
    fn cli_defaults_match_spec() {
        let a = build_args(&["--input", "notes"]);
        assert_eq!(a.db, PathBuf::from("index.db"));
        assert_eq!(a.strategy, "all");
        assert_eq!(
            (a.chunk_size, a.overlap, a.max_section, a.min_section),
            (1200, 200, 1500, 200)
        );
        assert_eq!(a.min_chars, 50_000);
        assert_eq!(a.unit, Unit::Chars);
        let e = &a.embed;
        assert_eq!(e.ollama_url, "http://localhost:11434");
        assert_eq!(e.model, "nomic-embed-text");
        assert_eq!(e.doc_prefix, "search_document: ");
        assert_eq!(e.query_prefix, "search_query: ");
        assert_eq!((e.batch, e.num_ctx, e.dim), (32, 2048, 768));
        let c = compare_args(&[]);
        assert_eq!(c.out, PathBuf::from("docs/chunking-comparison.md"));
    }

    #[tokio::test]
    async fn build_and_compare_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(BagOfWords { dim: 768 })
            .mount(&server)
            .await;
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("notes");
        std::fs::create_dir(&input).unwrap();
        write_corpus(&input);
        let db = tmp.path().join("index.db");
        let (input_s, db_s, url) = (input.to_str().unwrap(), db.to_str().unwrap(), server.uri());
        let flags = [
            "--input",
            input_s,
            "--db",
            db_s,
            "--ollama-url",
            &url,
            "--min-chars",
            "5000",
            "--chunk-size",
            "600",
            "--overlap",
            "100",
            "--max-section",
            "1500",
        ];
        let infos = run_build(&build_args(&flags)).await.unwrap();
        assert_eq!(
            infos
                .iter()
                .map(|i| i.strategy.as_str())
                .collect::<Vec<_>>(),
            ["fixed", "structure"]
        );

        let store = Store::open(&db).await.unwrap();
        assert_eq!(store.orphans().await.unwrap(), 0);
        let fixed = store.load("fixed").await.unwrap();
        let structure = store.load("structure").await.unwrap();
        assert!(
            fixed.len() > structure.len(),
            "{} vs {}",
            fixed.len(),
            structure.len()
        );
        for c in fixed.iter().chain(&structure) {
            assert_eq!(c.model, "nomic-embed-text");
            assert_eq!(c.vector.len(), 768);
            // Префикс задачи уходит только в модель.
            assert!(!c.row.text.starts_with("search_document"));
        }
        assert!(fixed.iter().any(|c| c.row.title == "Компьютерные сети"));
        assert!(structure.iter().any(|c| c.row.title == "ос"));
        assert!(structure.iter().any(|c| c.row.section == "Сети > UDP"));
        assert!(structure.iter().all(|c| !c.row.crosses_section));

        // Повторный build даёт те же chunk_id и не копит строки.
        let ids =
            |v: &[store::StoredChunk]| v.iter().map(|c| c.row.chunk_id.clone()).collect::<Vec<_>>();
        run_build(&build_args(&flags)).await.unwrap();
        assert_eq!(ids(&store.load("fixed").await.unwrap()), ids(&fixed));
        assert_eq!(
            ids(&store.load("structure").await.unwrap()),
            ids(&structure)
        );
        assert!(ids(&structure)[0].starts_with("structure:ос:0000"));

        let questions = tmp.path().join("questions.json");
        std::fs::write(
            &questions,
            r#"[
              {"question": "Как UDP отправляет датаграммы без подтверждений?", "expected_section": "Сети > UDP"},
              {"question": "Что делает планировщик процессов?", "expected_section": "Процессы"},
              {"question": "Как виртуальная память отображает страницы?", "expected_section": "Память"}
            ]"#,
        )
        .unwrap();
        let out = tmp.path().join("docs/report.md");
        let report = run_compare(&compare_args(&[
            "--db",
            db_s,
            "--questions",
            questions.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--ollama-url",
            &url,
        ]))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), report);
        assert!(
            report.contains("| Метрика | `fixed` | `structure` |"),
            "{report}"
        );
        assert!(report.contains("| MRR | 1.000 | 1.000 |"), "{report}");
        assert!(
            report.contains("baseline") && report.contains("rewrite+filter"),
            "{report}"
        );
        assert!(report.contains("Среднее кандидатов"), "{report}");
    }

    #[tokio::test]
    async fn context_overflow_names_the_chunk() {
        struct Limit;
        impl wiremock::Respond for Limit {
            fn respond(&self, request: &wiremock::Request) -> wiremock::ResponseTemplate {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                if body["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t.as_str().unwrap().len() > 3000)
                {
                    return wiremock::ResponseTemplate::new(400).set_body_json(
                        serde_json::json!({ "error": "the input length exceeds the context length" }),
                    );
                }
                BagOfWords { dim: 768 }.respond(request)
            }
        }
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(Limit)
            .mount(&server)
            .await;
        let tmp = tempfile::tempdir().unwrap();
        write_corpus(tmp.path());
        let db = tmp.path().join("index.db");
        let url = server.uri();
        let args = build_args(&[
            "--input",
            tmp.path().to_str().unwrap(),
            "--db",
            db.to_str().unwrap(),
            "--ollama-url",
            &url,
            "--min-chars",
            "0",
            "--strategy",
            "structure",
            "--max-section",
            "3000",
        ]);
        let err = format!("{:#}", run_build(&args).await.unwrap_err());
        assert!(err.contains("чанк structure:"), "{err}");
        assert!(err.contains("уменьшите --max-section"), "{err}");
    }

    async fn ollama_with_vocab() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(BagOfWords { dim: 768 })
            .mount(&server)
            .await;
        Mock::given(path("/api/show"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "model_info": {
                        "tokenizer.ggml.model": "bert",
                        "tokenizer.ggml.tokens": tokens::testutil::vocab(),
                    }
                })),
            )
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn sizes_in_tokens_bound_every_chunk() {
        let server = ollama_with_vocab().await;
        let tmp = tempfile::tempdir().unwrap();
        write_corpus(tmp.path());
        let db = tmp.path().join("index.db");
        let url = server.uri();
        let args = build_args(&[
            "--input",
            tmp.path().to_str().unwrap(),
            "--db",
            db.to_str().unwrap(),
            "--ollama-url",
            &url,
            "--min-chars",
            "0",
            "--unit",
            "tokens",
            "--chunk-size",
            "60",
            "--overlap",
            "10",
            "--max-section",
            "90",
            "--min-section",
            "15",
        ]);
        let infos = run_build(&args).await.unwrap();
        assert!(
            infos
                .iter()
                .all(|i| i.params.contains(r#""unit":"tokens""#)),
            "{infos:?}"
        );
        let wp = tokens::WordPiece::from_ollama(&url, "nomic-embed-text")
            .await
            .unwrap();
        let store = Store::open(&db).await.unwrap();
        let fixed = store.load("fixed").await.unwrap();
        let structure = store.load("structure").await.unwrap();
        for c in &fixed {
            assert!(
                wp.count(&c.row.text) <= 60,
                "{}: {}",
                c.row.chunk_id,
                wp.count(&c.row.text)
            );
        }
        for c in &structure {
            assert!(
                wp.count(&c.row.text) <= 90,
                "{}: {}",
                c.row.chunk_id,
                wp.count(&c.row.text)
            );
        }
        // Окна по 60 токенов длиннее одного токена на слово: чанков заметно больше десятка.
        assert!(fixed.len() > 10, "{}", fixed.len());
    }

    #[tokio::test]
    async fn token_sizes_must_fit_num_ctx() {
        let server = ollama_with_vocab().await;
        let tmp = tempfile::tempdir().unwrap();
        write_corpus(tmp.path());
        let url = server.uri();
        let args = build_args(&[
            "--input",
            tmp.path().to_str().unwrap(),
            "--ollama-url",
            &url,
            "--min-chars",
            "0",
            "--unit",
            "tokens",
            "--max-section",
            "2048",
        ]);
        let err = run_build(&args).await.unwrap_err().to_string();
        assert!(err.contains("не помещается в --num-ctx 2048"), "{err}");
    }

    #[tokio::test]
    async fn small_corpus_is_an_error_with_char_count() {
        let tmp = tempfile::tempdir().unwrap();
        write_docx(&tmp.path().join("a.docx"), "", &para("Коротко."), None);
        let args = build_args(&["--input", tmp.path().to_str().unwrap(), "--db", "unused.db"]);
        let err = run_build(&args).await.unwrap_err().to_string();
        assert!(
            err.contains("корпус слишком мал: 8 символов в 1 файлах"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn duplicate_file_stems_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("b")).unwrap();
        write_docx(&tmp.path().join("a.docx"), "", &para("x"), None);
        write_docx(&tmp.path().join("b/a.docx"), "", &para("y"), None);
        let args = build_args(&["--input", tmp.path().to_str().unwrap(), "--min-chars", "0"]);
        let err = run_build(&args).await.unwrap_err().to_string();
        assert!(err.contains("одинаковое имя файла"), "{err}");
    }

    #[tokio::test]
    async fn compare_refuses_other_model() {
        let server = MockServer::start().await;
        Mock::given(path("/api/embed"))
            .respond_with(BagOfWords { dim: 768 })
            .mount(&server)
            .await;
        let tmp = tempfile::tempdir().unwrap();
        write_corpus(tmp.path());
        let db = tmp.path().join("index.db");
        let (dir, db_s, url) = (
            tmp.path().to_str().unwrap(),
            db.to_str().unwrap(),
            server.uri(),
        );
        run_build(&build_args(&[
            "--input",
            dir,
            "--db",
            db_s,
            "--ollama-url",
            &url,
            "--min-chars",
            "0",
        ]))
        .await
        .unwrap();
        let err = run_compare(&compare_args(&[
            "--db",
            db_s,
            "--ollama-url",
            &url,
            "--model",
            "bge-m3",
        ]))
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("построена моделью nomic-embed-text"), "{err}");
    }

    /// Живой прогон против настоящего Ollama с `nomic-embed-text`:
    /// `INDEX_MCP_OLLAMA_URL=http://localhost:11434 cargo test live_ollama -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn live_ollama_embeds_with_prefixes() {
        let url = std::env::var("INDEX_MCP_OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".into());
        let url_for_show = url.clone();
        let e = Embedder::new(EmbedConfig {
            url,
            model: "nomic-embed-text".into(),
            batch: 32,
            num_ctx: 2048,
            dim: 768,
        })
        .unwrap();
        let docs = e
            .embed(
                "search_document: ",
                &[
                    "Протокол TCP гарантирует доставку байтов по порядку.",
                    "Рецепт борща: свёкла, капуста, картофель.",
                ],
            )
            .await
            .unwrap();
        let q = e
            .embed(
                "search_query: ",
                &["Как TCP обеспечивает надёжную доставку?"],
            )
            .await
            .unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&q[0], &docs[0]) > dot(&q[0], &docs[1]));
        // Наш счёт токенов сверяется со счётом самого Ollama.
        let wp = tokens::WordPiece::from_ollama(&url_for_show, "nomic-embed-text")
            .await
            .unwrap();
        let sample = "Протокол TCP устанавливает соединение, а UDP — нет. Ёмкость окна растёт.";
        let body: serde_json::Value = reqwest::Client::new()
            .post(format!("{url_for_show}/api/embed"))
            .json(&serde_json::json!({ "model": "nomic-embed-text", "input": sample }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let ollama = body["prompt_eval_count"].as_u64().unwrap() as usize;
        let ours = wp.count(sample) + 2; // [CLS] и [SEP]
        assert!(
            ours.abs_diff(ollama) <= 2,
            "наш счёт {ours}, Ollama {ollama}"
        );
        // Чанк длиной в умолчание --max-section (1500 символов русского текста)
        // помещается в контекст модели в 2048 токенов.
        let long = "Раздел конспекта про сети. ".repeat(55);
        e.embed("search_document: ", &[long.as_str()])
            .await
            .unwrap();
    }
}
