-- Чанки всех стратегий в одной таблице: сравнение идёт по колонке strategy.
CREATE TABLE chunks (
    chunk_id         TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL,
    section          TEXT NOT NULL,
    ordinal          INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    text             TEXT NOT NULL,
    -- Метрики границ считаются при построении: после него структуры
    -- документа (заголовков и абзацев) в базе уже нет.
    crosses_section  INTEGER NOT NULL,
    cut_mid_sentence INTEGER NOT NULL,
    UNIQUE (chunk_id)
);
CREATE INDEX chunks_strategy ON chunks (strategy);

-- Вектор — f32 little-endian подряд; model и dim хранятся при каждом
-- векторе, чтобы смесь моделей в одной базе была видна запросом.
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

-- Последнее построение каждой стратегии: объём корпуса и время эмбеддинга.
CREATE TABLE builds (
    strategy  TEXT PRIMARY KEY NOT NULL,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    files     INTEGER NOT NULL,
    chars     INTEGER NOT NULL,
    chunks    INTEGER NOT NULL,
    embed_ms  INTEGER NOT NULL,
    params    TEXT NOT NULL,
    built_at  TEXT NOT NULL
);
