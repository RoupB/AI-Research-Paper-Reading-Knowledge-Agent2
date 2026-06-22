-- db/schema.sql

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=OFF;   -- FK declared but not enforced (SQLite default)

-- ── Papers ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    authors             TEXT NOT NULL,          -- JSON array
    abstract            TEXT NOT NULL,
    published           TEXT NOT NULL,          -- ISO-8601
    pdf_url             TEXT NOT NULL,
    arxiv_url           TEXT NOT NULL,
    lora_variant_tag    TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'discovered',
    repo_url            TEXT,
    repo_confidence     REAL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- ── Benchmark Claims ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS benchmark_claims (
    claim_id            TEXT PRIMARY KEY,       -- UUID
    paper_id            TEXT NOT NULL REFERENCES papers(arxiv_id),
    metric              TEXT NOT NULL,
    dataset             TEXT NOT NULL,
    model_base          TEXT NOT NULL,
    reported_value      REAL NOT NULL,
    unit                TEXT,
    conditions          TEXT NOT NULL DEFAULT '{}',   -- JSON dict
    is_conditional      INTEGER NOT NULL DEFAULT 0,  -- bool
    source_section      TEXT NOT NULL,
    raw_text            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_paper ON benchmark_claims(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_metric_dataset ON benchmark_claims(metric, dataset, model_base);

-- ── Code Facts ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS code_facts (
    fact_id             TEXT PRIMARY KEY,       -- UUID
    paper_id            TEXT NOT NULL REFERENCES papers(arxiv_id),
    repo_url            TEXT NOT NULL,
    fact_type           TEXT NOT NULL,          -- hyperparameter|dataset|metric_logged|missing_eval
    key                 TEXT NOT NULL,
    value               TEXT,
    file_path           TEXT NOT NULL,
    line_range          TEXT,                   -- "start,end" or NULL
    evidence            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_paper ON code_facts(paper_id);

-- ── Reproducibility Gaps ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reproducibility_gaps (
    gap_id              TEXT PRIMARY KEY,       -- UUID
    paper_id            TEXT NOT NULL REFERENCES papers(arxiv_id),
    claim_id            TEXT NOT NULL REFERENCES benchmark_claims(claim_id),
    fact_id             TEXT REFERENCES code_facts(fact_id),
    gap_type            TEXT NOT NULL,          -- missing_code|value_mismatch|dataset_mismatch|condition_undisclosed|metric_not_implemented
    severity            TEXT NOT NULL,          -- critical|major|minor
    description         TEXT NOT NULL,
    paper_value         TEXT,
    code_value          TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gaps_paper ON reproducibility_gaps(paper_id);
CREATE INDEX IF NOT EXISTS idx_gaps_severity ON reproducibility_gaps(severity);

-- ── Contradictions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id    TEXT PRIMARY KEY,       -- UUID
    paper_a_id          TEXT NOT NULL REFERENCES papers(arxiv_id),
    paper_b_id          TEXT NOT NULL REFERENCES papers(arxiv_id),
    claim_a_id          TEXT NOT NULL REFERENCES benchmark_claims(claim_id),
    claim_b_id          TEXT NOT NULL REFERENCES benchmark_claims(claim_id),
    contradiction_type  TEXT NOT NULL,          -- direct_numeric|conditional_flip|dataset_scope
    description         TEXT NOT NULL,
    severity            TEXT NOT NULL,          -- high|medium|low
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contradictions_papers ON contradictions(paper_a_id, paper_b_id);

-- ── MCP run management ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    research_question   TEXT,
    variants            TEXT,                   -- JSON array
    max_papers          INTEGER,
    status              TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|error
    current_stage       TEXT,
    progress            REAL DEFAULT 0.0,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_tool_calls (
    call_id             TEXT PRIMARY KEY,
    run_id              TEXT,
    tool_name           TEXT NOT NULL,
    request             TEXT,                   -- redacted JSON
    response            TEXT,                   -- redacted JSON
    status              TEXT NOT NULL,          -- success|error
    latency_ms          REAL,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcp_calls_run ON mcp_tool_calls(run_id);
