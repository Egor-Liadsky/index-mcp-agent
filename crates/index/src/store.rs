//! SQLite-индекс: чанки, их векторы и сведения о построениях.
//!
//! Векторы лежат в той же базе, что и тексты, BLOB-ом f32 little-endian:
//! на корпусе конспектов это сотни чанков, и перебор косинусом по всем
//! векторам в памяти быстрее любой подготовки ANN-индекса.

use anyhow::{Context, Result, ensure};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::Path;

/// Строка таблицы `chunks`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRow {
    pub chunk_id: String,
    pub strategy: String,
    pub source: String,
    pub title: String,
    pub section: String,
    pub ordinal: i64,
    pub char_start: i64,
    pub char_end: i64,
    pub text: String,
    pub crosses_section: bool,
    pub cut_mid_sentence: bool,
}

/// Чанк вместе с вектором — то, что читает `compare`.
#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub row: ChunkRow,
    pub model: String,
    pub vector: Vec<f32>,
}

/// Строка таблицы `builds`.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildInfo {
    pub strategy: String,
    pub model: String,
    pub dim: i64,
    pub files: i64,
    pub chars: i64,
    pub chunks: i64,
    pub embed_ms: i64,
    pub params: String,
}

pub struct Store {
    pool: SqlitePool,
}

pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len().is_multiple_of(4),
        "длина BLOB вектора {} не кратна 4",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

impl Store {
    /// Открывает (или создаёт) базу и применяет миграции.
    pub async fn open(path: &Path) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .with_context(|| format!("не открыть базу {}", path.display()))?;
        sqlx::migrate!("../../migrations")
            .run(&pool)
            .await
            .context("не применить миграции")?;
        Ok(Self { pool })
    }

    /// Заменяет все строки стратегии одной транзакцией: при ошибке в базе
    /// остаётся прежний индекс стратегии, а не половина нового.
    pub async fn replace_strategy(
        &self,
        info: &BuildInfo,
        rows: &[ChunkRow],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(
            rows.len() == vectors.len(),
            "чанков {} и векторов {}",
            rows.len(),
            vectors.len()
        );
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE strategy = ?)",
        )
        .bind(&info.strategy)
        .execute(&mut *tx)
        .await?;
        sqlx::query("DELETE FROM chunks WHERE strategy = ?")
            .bind(&info.strategy)
            .execute(&mut *tx)
            .await?;
        for (row, vector) in rows.iter().zip(vectors) {
            ensure!(
                row.strategy == info.strategy,
                "чанк {} чужой стратегии",
                row.chunk_id
            );
            sqlx::query(
                "INSERT INTO chunks (chunk_id, strategy, source, title, section, ordinal, char_start, \
                 char_end, text, crosses_section, cut_mid_sentence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&row.chunk_id)
            .bind(&row.strategy)
            .bind(&row.source)
            .bind(&row.title)
            .bind(&row.section)
            .bind(row.ordinal)
            .bind(row.char_start)
            .bind(row.char_end)
            .bind(&row.text)
            .bind(row.crosses_section)
            .bind(row.cut_mid_sentence)
            .execute(&mut *tx)
            .await
            .with_context(|| format!("не записать чанк {}", row.chunk_id))?;
            sqlx::query(
                "INSERT INTO embeddings (chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
            )
            .bind(&row.chunk_id)
            .bind(&info.model)
            .bind(vector.len() as i64)
            .bind(encode_vector(vector))
            .execute(&mut *tx)
            .await?;
        }
        sqlx::query(
            "INSERT OR REPLACE INTO builds (strategy, model, dim, files, chars, chunks, embed_ms, params, built_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&info.strategy)
        .bind(&info.model)
        .bind(info.dim)
        .bind(info.files)
        .bind(info.chars)
        .bind(info.chunks)
        .bind(info.embed_ms)
        .bind(&info.params)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn builds(&self) -> Result<Vec<BuildInfo>> {
        let rows = sqlx::query(
            "SELECT strategy, model, dim, files, chars, chunks, embed_ms, params FROM builds ORDER BY strategy",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| BuildInfo {
                strategy: r.get(0),
                model: r.get(1),
                dim: r.get(2),
                files: r.get(3),
                chars: r.get(4),
                chunks: r.get(5),
                embed_ms: r.get(6),
                params: r.get(7),
            })
            .collect())
    }

    /// Чанки стратегии с векторами в порядке файлов и `ordinal`.
    pub async fn load(&self, strategy: &str) -> Result<Vec<StoredChunk>> {
        let rows = sqlx::query(
            "SELECT c.chunk_id, c.strategy, c.source, c.title, c.section, c.ordinal, c.char_start, \
             c.char_end, c.text, c.crosses_section, c.cut_mid_sentence, e.model, e.vector \
             FROM chunks c JOIN embeddings e USING (chunk_id) WHERE c.strategy = ? \
             ORDER BY c.source, c.ordinal",
        )
        .bind(strategy)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|r| {
                Ok(StoredChunk {
                    row: ChunkRow {
                        chunk_id: r.get(0),
                        strategy: r.get(1),
                        source: r.get(2),
                        title: r.get(3),
                        section: r.get(4),
                        ordinal: r.get(5),
                        char_start: r.get(6),
                        char_end: r.get(7),
                        text: r.get(8),
                        crosses_section: r.get(9),
                        cut_mid_sentence: r.get(10),
                    },
                    model: r.get(11),
                    vector: decode_vector(r.get::<&[u8], _>(12))?,
                })
            })
            .collect()
    }

    /// Чанки без вектора — после корректного `build` их ноль.
    #[cfg(test)]
    pub async fn orphans(&self) -> Result<i64> {
        Ok(sqlx::query_scalar(
            "SELECT count(*) FROM chunks c LEFT JOIN embeddings e USING (chunk_id) WHERE e.chunk_id IS NULL",
        )
        .fetch_one(&self.pool)
        .await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(strategy: &str, n: i64) -> ChunkRow {
        ChunkRow {
            chunk_id: format!("{strategy}:a:{n:04}"),
            strategy: strategy.into(),
            source: "a.docx".into(),
            title: "A".into(),
            section: "Раздел".into(),
            ordinal: n,
            char_start: n * 10,
            char_end: n * 10 + 5,
            text: format!("текст {n}"),
            crosses_section: n % 2 == 0,
            cut_mid_sentence: false,
        }
    }

    fn info(strategy: &str, chunks: i64) -> BuildInfo {
        BuildInfo {
            strategy: strategy.into(),
            model: "nomic-embed-text".into(),
            dim: 2,
            files: 1,
            chars: 100,
            chunks,
            embed_ms: 7,
            params: "{}".into(),
        }
    }

    #[tokio::test]
    async fn replace_keeps_other_strategies_and_roundtrips_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.db")).await.unwrap();
        let v = vec![vec![0.6f32, 0.8]; 3];
        store
            .replace_strategy(
                &info("fixed", 3),
                &[row("fixed", 0), row("fixed", 1), row("fixed", 2)],
                &v,
            )
            .await
            .unwrap();
        store
            .replace_strategy(&info("structure", 1), &[row("structure", 0)], &v[..1])
            .await
            .unwrap();
        // Повторное построение fixed с меньшим числом чанков убирает старые строки.
        store
            .replace_strategy(&info("fixed", 1), &[row("fixed", 0)], &v[..1])
            .await
            .unwrap();
        let fixed = store.load("fixed").await.unwrap();
        assert_eq!(fixed.len(), 1);
        assert_eq!(fixed[0].row, row("fixed", 0));
        assert_eq!(fixed[0].vector, vec![0.6, 0.8]);
        assert_eq!(store.load("structure").await.unwrap().len(), 1);
        assert_eq!(store.orphans().await.unwrap(), 0);
        let builds = store.builds().await.unwrap();
        assert_eq!(builds, vec![info("fixed", 1), info("structure", 1)]);
    }

    #[tokio::test]
    async fn duplicate_chunk_id_rolls_back_whole_strategy() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.db")).await.unwrap();
        let v = vec![vec![1.0f32, 0.0]; 2];
        store
            .replace_strategy(&info("fixed", 1), &[row("fixed", 5)], &v[..1])
            .await
            .unwrap();
        let err = store
            .replace_strategy(&info("fixed", 2), &[row("fixed", 0), row("fixed", 0)], &v)
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("fixed:a:0000"), "{err:#}");
        let kept = store.load("fixed").await.unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].row.chunk_id, "fixed:a:0005");
    }

    #[test]
    fn vector_encoding_is_little_endian_f32() {
        assert_eq!(encode_vector(&[1.0]), vec![0, 0, 0x80, 0x3f]);
        assert_eq!(decode_vector(&[0, 0, 0x80, 0x3f]).unwrap(), vec![1.0]);
        assert!(decode_vector(&[1, 2, 3]).is_err());
    }
}
