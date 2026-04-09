CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS drawers (
    id TEXT PRIMARY KEY,
    wing TEXT NOT NULL,
    room TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    source_file TEXT DEFAULT '',
    chunk_index INTEGER DEFAULT 0,
    added_by TEXT DEFAULT 'mempalace',
    filed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_drawers_wing ON drawers(wing);
CREATE INDEX IF NOT EXISTS idx_drawers_room ON drawers(room);
CREATE INDEX IF NOT EXISTS idx_drawers_source ON drawers(source_file);
-- HNSW params tuned for 400k+ corpora:
--   m = 32 (vs default 16): denser graph, higher recall at query time.
--   ef_construction = 200 (vs default 64): better graph quality at build time.
-- Default params produced recall@10 ≈ 0% on the mempalace corpus — HNSW
-- returned noise clusters while the true top-k stayed unreachable even at
-- hnsw.ef_search=1000. Also ensure maintenance_work_mem is ≥ 4GB and
-- max_parallel_maintenance_workers = 0 (docker /dev/shm is too small) when
-- reindexing. See reference_pgvector_hnsw_tuning for the full recipe.
CREATE INDEX IF NOT EXISTS idx_drawers_embedding ON drawers
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 200);

CREATE TABLE IF NOT EXISTS compressed (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'unknown',
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object TEXT NOT NULL REFERENCES entities(id),
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL DEFAULT 1.0,
    source_closet TEXT,
    source_file TEXT,
    extracted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);
