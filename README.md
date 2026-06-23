# ClaimCheck — LoRA Research Audit System

> **Automated reproducibility and cross-paper contradiction auditing for LoRA-variant fine-tuning papers**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Claude Sonnet](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.6-blueviolet.svg)](https://www.anthropic.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/database-SQLite-green.svg)](https://sqlite.org)
[![Tests](https://img.shields.io/badge/tests-pytest-yellowgreen.svg)](https://pytest.org)

---

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [System Architecture](#system-architecture)
4. [Agent Pipeline](#agent-pipeline)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Configuration](#configuration)
9. [Running the Pipeline](#running-the-pipeline)
10. [Streamlit UI](#streamlit-ui)
11. [MCP Server](#mcp-server)
12. [Data Models](#data-models)
13. [Testing](#testing)
14. [API Cost Estimate](#api-cost-estimate)
15. [Validation & Evaluation](#validation--evaluation)
16. [Limitations](#limitations)
17. [Ethical Considerations](#ethical-considerations)
18. [Future Work](#future-work)

---

## Overview

Published PEFT papers (LoRA, QLoRA, DoRA, AdaLoRA, LoRA+, VeRA, …) make numerous claims about improvements over baseline LoRA on shared benchmarks. In practice:

- **Claims are not independently verifiable** — hyperparameters differ, configs are incomplete, or released code does not match the described method.
- **Claims contradict each other** — "beats LoRA by X%" means different things when conditions (rank, dataset, model size) differ across papers.
- **No systematic tool exists** to map where the literature agrees, disagrees, or is only conditionally comparable.

**ClaimCheck** is a 7-agent agentic pipeline that autonomously:

| Step | What it does |
|---|---|
| Discovers | Finds LoRA-variant papers on arXiv via a refinement loop |
| Resolves | Locates the official GitHub repository for each paper |
| Extracts | Parses PDFs → structured `BenchmarkClaim` records |
| Audits | Reads repository code → `CodeFact` records |
| Gaps | Reconciles paper claims vs. code → `ReproducibilityGap` records |
| Contradictions | Two-pass (heuristic + LLM) cross-paper conflict detection with synonym normalisation |
| Reports | Generates Markdown + HTML audit reports |

All findings are persisted in a **SQLite** database and surfaced through an interactive **Streamlit** web application.

---

## Research Questions

| # | Research Question | Hypothesis |
|---|---|---|
| **RQ1** | What fraction of benchmark claims in LoRA-variant papers can be reproduced from the released code? | H1: < 60% of critical claims are directly reproducible (hyperparameters, datasets, eval code all present and matching) |
| **RQ2** | How frequently do cross-paper benchmark comparisons contradict each other on the same metric/dataset/base-model triple? | H2: ≥ 30% of same-setup comparisons show a direct numeric contradiction (> 2% relative difference) |
| **RQ3** | How accurately can LLM agents extract structured benchmark claims from ML paper PDFs, compared to human annotation? | H3: Claim extraction achieves ≥ 0.80 F1 against human-annotated ground truth on a 25-paper validation set |

These hypotheses are validated against a **hand-annotated validation set** (20–30 papers) stored in `tests/fixtures/validation_set.json`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Entry Points                                   │
│   python main.py run          │         streamlit run app/streamlit_app.py │
└──────────────┬────────────────┴──────────────────────────────────────────┘
               │ Interactive session (UserSessionOutput)
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       LangGraph Pipeline (main.py)                       │
│                                                                          │
│  node_discover → node_resolve_repos → node_extract_claims               │
│       → node_analyze_code → node_gap_analysis                           │
│       → node_contradiction_mapping → node_report                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
       SQLite audit.db    artifacts/         LangGraph
       (5 core tables +   (PDF cache,        checkpoint
        2 MCP tables)      claim graph,       (SqliteSaver)
                           reports)
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Streamlit Web App (app/)                           │
│   Home Dashboard · Search · Claims · Gaps · Contradictions · Report     │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MCP Server (mcp_server/)                         │
│   server.py · schemas.py · auth.py · run_manager.py · tools/*.py        │
│                       ↕ services.py (shared service layer)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

### Agent 1 — PaperDiscoveryAgent
**File:** `agents/paper_discovery.py`

Searches arXiv using query terms from the user session. Runs as a **refinement loop** (up to 3 rounds) — widening the search if fewer than `MIN_PAPERS_THRESHOLD` relevant papers are found. Uses an LLM relevance filter to tag each paper with its LoRA variant.

### Agent 2 — RepoResolutionAgent
**File:** `agents/repo_resolution.py`

For each discovered paper, locates the official GitHub repository by scanning the PDF first pages and falling back to the GitHub search API. Papers with `confidence < 0.5` have their repo set to `null` and are skipped by downstream agents.

### Agent 3 — ClaimExtractionAgent
**File:** `agents/claim_extraction.py`

Reads the full paper PDF (via `pdfplumber` / `pymupdf` fallback) and extracts every quantitative benchmark claim into structured `BenchmarkClaim` records: metric, dataset, base model, value, unit, conditions, and the verbatim source sentence.

### Agent 4 — CodeAnalysisAgent
**File:** `agents/code_analysis.py`

Fetches the GitHub repository tree and reads relevant files (`train*.py`, `run*.py`, `config*.yaml`, etc.) to extract `CodeFact` records — hardcoded hyperparameters, dataset loading paths, metrics logged, and eval scripts present or absent.

### Agent 5 — GapAnalysisAgent
**File:** `agents/gap_analysis.py`

Compares the paper's `BenchmarkClaim` records against its `CodeFact` records to produce `ReproducibilityGap` records. Gap types: `missing_code`, `value_mismatch`, `dataset_mismatch`, `condition_undisclosed`, `metric_not_implemented`. Severity: `critical`, `major`, `minor`.

### Agent 6 — ContradictionMappingAgent
**File:** `agents/contradiction_mapping.py`

Corpus-level agent (no per-paper input — loads everything from the DB). Uses a **two-pass approach**:

| Pass | Method | Detects |
|---|---|---|
| **Pass 1 — Heuristic** | Rule-based, no LLM | `direct_numeric`: flags any cross-paper pair with ≥ 1% relative diff or ≥ 0.5 absolute diff |
| **Pass 2 — LLM** | `call_llm_json()` per cluster | `conditional_flip`, `dataset_scope`, numeric misses from Pass 1 |

Before clustering, metric/dataset/model strings are **normalised through synonym tables** (e.g. `"acc"→"accuracy"`, `"sst2"→"sst-2"`, `"llama-7b-hf"→"llama-7b"`) to prevent false split-clusters from spelling variants. A `seen_pairs` frozenset deduplicates findings across both passes.

The resulting **NetworkX DiGraph** (nodes = claims, edges = contradictions) is saved to `artifacts/claim_graph.json`.

**Severity thresholds:** `high` ≥ 5% rel diff or rank flip · `medium` 1–5% · `low` subtle discrepancy

### Agent 7 — ReportGenerationAgent
**File:** `agents/report_generation.py`

Aggregates all DB records and produces dual-format reports:
- `reports/audit_report.md` — human-readable Markdown
- `reports/audit_report.html` — rendered HTML via Jinja2 template

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | Async support, rich ML ecosystem |
| LLM SDK | `anthropic` (claude-sonnet-4-6) | Structured JSON extraction, long-context PDF analysis |
| LLM helpers | `agents/llm.py` | `make_client()`, `call_llm_json()` with mandatory retry pattern |
| Agent orchestration | **LangGraph** | Checkpointed state graph, conditional edges, agentic loops |
| Service layer | `services.py` | Shared business logic for pipeline and MCP |
| ArXiv access | `arxiv` PyPI + `httpx` | Paper metadata + PDF download |
| GitHub access | `PyGithub` + `httpx` | Repo tree traversal and file fetch |
| PDF parsing | `pdfplumber` + `pymupdf` | Table extraction, section detection |
| Database | `aiosqlite` (SQLite) | Lightweight, file-portable, WAL mode |
| Claim graph | `networkx` | Cross-paper contradiction graph |
| Data validation | `pydantic` v2 | Schema enforcement at every agent boundary |
| Configuration | `pydantic-settings` + `.env` | Typed, validated secrets at startup |
| CLI | `typer` | `run` / `ui` commands |
| Frontend | `streamlit` | Interactive web UI |
| Charts | `plotly` | Reproducibility and contradiction visualizations |
| Reporting | `jinja2` + `markdown` | HTML and Markdown report generation |
| Logging | `structlog` | JSON-structured per-agent logs |
| Testing | `pytest` + `pytest-asyncio` | Unit + integration tests |
| Interoperability | **MCP server** | Standardized tool interface for external clients |
| Deployment | Streamlit Community Cloud | Free public hosting, auto-deploy from GitHub |

---

## Project Structure

```
lora-audit/
├── .env                          # Local secrets (gitignored)
├── .env.example                  # Committed template
├── config.py                     # pydantic-settings config loader
├── models.py                     # All Pydantic v2 data models
├── main.py                       # LangGraph graph + typer CLI
├── services.py                   # Service layer (shared by pipeline + MCP)
│
├── session/
│   └── user_session.py           # Interactive + non-interactive scoping
│
├── agents/
│   ├── base_agent.py             # run_with_limit, with_retry, get_logger
│   ├── llm.py                    # make_client(), call_llm(), call_llm_json()
│   ├── paper_discovery.py        # Agent 1
│   ├── repo_resolution.py        # Agent 2
│   ├── claim_extraction.py       # Agent 3
│   ├── code_analysis.py          # Agent 4
│   ├── gap_analysis.py           # Agent 5
│   ├── contradiction_mapping.py  # Agent 6 (two-pass, synonym normalisation)
│   └── report_generation.py      # Agent 7
│
├── tools/
│   ├── arxiv_tool.py             # search_arxiv(), fetch_paper_text()
│   ├── github_tool.py            # resolve_repo(), fetch_repo_tree(), fetch_file()
│   └── pdf_tool.py               # SYNC: fetch_and_cache_pdf(), parse_tables(), extract_sections()
│
├── db/
│   ├── schema.sql                # Table definitions (5 core + 2 MCP tables)
│   ├── init_db.py                # Database initialisation
│   └── queries.py                # All async DB operations (no SQL in agents)
│
├── mcp_server/
│   ├── server.py                 # MCP server bootstrap
│   ├── schemas.py                # Request/response Pydantic models
│   ├── auth.py                   # Token auth + rate limiting
│   ├── run_manager.py            # Run lifecycle helpers
│   └── tools/
│       ├── pipeline.py           # start_pipeline_run, get_run_status, cancel_run
│       ├── papers.py             # discover_papers, resolve_repos
│       ├── claims.py             # extract_claims, list_claims
│       ├── code.py               # analyze_code
│       ├── gaps.py               # analyze_gaps, list_gaps
│       ├── contradictions.py     # map_contradictions, list_contradictions
│       └── report.py             # generate_report
│
├── app/
│   ├── streamlit_app.py          # Home dashboard (KPI metrics)
│   ├── pages/
│   │   ├── 01_Search.py          # Trigger pipeline + monitor progress
│   │   ├── 02_Claims.py          # Browse claims + manual annotation
│   │   ├── 03_Gaps.py            # Reproducibility gap viewer
│   │   ├── 04_Contradictions.py  # Contradiction network graph
│   │   └── 05_Report.py          # Download audit report
│   └── components/
│       ├── charts.py             # Plotly chart helpers (return go.Figure only)
│       └── db_reader.py          # Read-only sync SQLite wrappers for Streamlit
│
├── templates/
│   ├── report.md.j2              # Markdown report template
│   └── report.html.j2            # HTML report template
│
├── tests/
│   ├── conftest.py               # Shared fixtures (test_db, mock_llm, sample_paper)
│   ├── test_db.py
│   ├── test_user_session.py
│   ├── test_arxiv_tool.py
│   ├── test_github_tool.py
│   ├── test_pdf_tool.py
│   ├── test_paper_discovery.py
│   ├── test_repo_resolution.py
│   ├── test_claim_extraction.py
│   ├── test_code_analysis.py
│   ├── test_gap_analysis.py
│   ├── test_contradiction_mapping.py
│   ├── test_report_generation.py
│   ├── test_mcp.py
│   ├── test_pipeline_smoke.py
│   └── test_validation_set.py
│
├── data/
│   └── audit.db                  # Created at runtime (gitignored)
├── artifacts/
│   ├── pdfs/                     # PDF text cache
│   └── claim_graph.json          # NetworkX node-link graph (written by Agent 6)
├── logs/
└── reports/                      # Generated audit reports
```

---

## Quick Start

### Prerequisites

- Python 3.11
- An [Anthropic API key](https://console.anthropic.com/)
- A [GitHub Personal Access Token](https://github.com/settings/tokens) with `read:public_repo` scope

### 1. Clone and install

```bash
git clone <repo-url>
cd lora-audit
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and GITHUB_TOKEN
```

### 3. Initialise the database

```bash
python db/init_db.py
```

### 4. Verify the setup

```bash
python -c "from config import settings; print('Config OK:', settings.anthropic_model)"
```

---

## Configuration

All settings are loaded from `.env` via `config.py` (pydantic-settings). Missing required keys raise `ValidationError` at startup.

```dotenv
# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MAX_TOKENS=8192
ANTHROPIC_TEMPERATURE=0.1          # low for structured extraction

# ── GitHub ────────────────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_...               # read:public_repo scope

# ── ArXiv ─────────────────────────────────────────────────────────────────
ARXIV_MAX_RESULTS_PER_QUERY=50
ARXIV_RATE_LIMIT_SLEEP=1.0

# ── Pipeline ──────────────────────────────────────────────────────────────
PIPELINE_MAX_PAPERS=100
PIPELINE_CONCURRENCY=3             # max parallel agent tasks
SKIP_PAPERS_WITHOUT_REPO=true

# ── Storage ───────────────────────────────────────────────────────────────
DB_PATH=./data/audit.db
ARTIFACTS_DIR=./artifacts
REPORT_OUTPUT_DIR=./reports
PDF_CACHE_DIR=./artifacts/pdfs

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL=INFO                     # DEBUG | INFO | WARNING | ERROR

# ── MCP (optional) ────────────────────────────────────────────────────────
MCP_ENABLED=true
MCP_TRANSPORT=stdio
MCP_AUTH_TOKEN=
MCP_RATE_LIMIT_PER_MIN=60
```

---

## Running the Pipeline

### Interactive CLI (recommended)

```bash
python main.py run --max 30 --output reports/
```

An interactive session starts automatically, asking for your research question (e.g. *"Which LoRA variants work best for instruction tuning of LLaMA models?"*). The session uses one LLM call to extract search queries and a clarifying question if needed.

### Resume from last checkpoint

```bash
python main.py run --max 30 --resume
```

LangGraph's `SqliteSaver` checkpointer means interrupted runs restart from the last completed node — no re-processing of already-completed papers.

### Debug mode

```bash
LOG_LEVEL=DEBUG python main.py run --max 5
```

### Non-interactive (scripting)

Use `run_user_session_from_text()` programmatically (bypasses stdin — used by the Streamlit UI):

```python
from session.user_session import run_user_session_from_text

session = await run_user_session_from_text(
    "QLoRA vs LoRA on MT-Bench",
    variants=["QLoRA", "LoRA"],
)
```

---

## Streamlit UI

```bash
python main.py ui
# or directly:
streamlit run app/streamlit_app.py
```

| Page | Path | What it shows |
|---|---|---|
| **Home** | `/` | KPI metrics: papers audited, claims, gaps, contradictions |
| **Search** | `01_Search` | Trigger a new pipeline run, monitor progress |
| **Claims** | `02_Claims` | Filter/browse all `BenchmarkClaim` records; add manual annotations |
| **Gaps** | `03_Gaps` | Reproducibility gap analysis with severity breakdown per paper |
| **Contradictions** | `04_Contradictions` | Force-directed contradiction network graph |
| **Report** | `05_Report` | Download the full audit report (MD or HTML) |

---

## MCP Server

ClaimCheck exposes a **Model Context Protocol** server for external integrations. The same `services.py` layer is used — no business logic is duplicated.

### Start the server

```bash
# stdio transport (default)
python -m mcp_server.server

# HTTP transport
MCP_TRANSPORT=http MCP_PORT=8765 python -m mcp_server.server
```

### Available tools

| Tool | Mutating | Description |
|---|---|---|
| `start_pipeline_run` | Yes | Queue a new audit run |
| `get_run_status` | No | Poll progress by `run_id` |
| `cancel_run` | Yes | Cancel a queued or running pipeline |
| `discover_papers` | Yes | Run paper discovery for a run |
| `extract_claims` | Yes | Run claim extraction for one paper |
| `analyze_code` | Yes | Run code analysis for one paper |
| `analyze_gaps` | Yes | Run gap analysis for one paper |
| `map_contradictions` | Yes | Run corpus-level contradiction mapping |
| `generate_report` | Yes | Generate the audit report |
| `list_claims` / `list_gaps` / `list_contradictions` | No | Query findings |

Mutating tools require the `MCP_AUTH_TOKEN` header. All tool calls are logged to the `mcp_tool_calls` table with latency and status.

---

## Data Models

All models are defined in `models.py` (Pydantic v2). Every agent validates its input and output against these schemas.

### Core Models

```
Paper                 arxiv_id, title, authors, abstract, pdf_url, repo_url,
                      lora_variant_tag, status (DISCOVERED→DONE|FAILED),
                      citation_count (used as pivot weight for large clusters)

BenchmarkClaim        paper_id, metric, dataset, model_base, reported_value,
                      unit, conditions{}, is_conditional, claim_confidence,
                      source_section, raw_text

CodeFact              paper_id, repo_url, fact_type (hyperparameter|dataset|
                      metric_logged|missing_eval), key, value, file_path,
                      line_range, evidence

ReproducibilityGap    gap_id, paper_id, claim_id, fact_id, gap_type
                      (missing_code|value_mismatch|dataset_mismatch|
                       condition_undisclosed|metric_not_implemented),
                      severity (critical|major|minor), description,
                      paper_value, code_value

Contradiction         contradiction_id, paper_a_id, paper_b_id,
                      claim_a_id, claim_b_id,
                      contradiction_type (direct_numeric|conditional_flip|
                                          dataset_scope),
                      severity (high|medium|low), description
```

### Pipeline Status Progression

```
DISCOVERED → REPO_RESOLVED → CLAIMS_EXTRACTED → CODE_ANALYZED → GAPS_ANALYZED → DONE
                                                                               ↘ FAILED
```

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific module
pytest tests/test_contradiction_mapping.py -v

# Run with debug logging
LOG_LEVEL=DEBUG pytest tests/ -s

# Run the end-to-end smoke test (uses live APIs — requires .env)
pytest tests/test_pipeline_smoke.py -v -s
```

### Test infrastructure

- All tests use the `test_db` fixture (isolated in-memory SQLite per test) and `mock_llm` fixture (injectable `AsyncAnthropic` client returning fixture JSON) from `tests/conftest.py`.
- No test makes a live LLM call except `test_pipeline_smoke.py`.
- `test_contradiction_mapping.py` verifies: detection found, single-paper cluster skipped, empty corpus, and hallucinated claim IDs discarded (heuristic still fires).

---

## API Cost Estimate

For a **30-paper run** (default `PIPELINE_MAX_PAPERS`):

| Agent | Tokens / paper | × 30 papers | Subtotal |
|---|---|---|---|
| PaperDiscovery | ~2,000 | 30 | 60K |
| RepoResolution | ~4,000 | 30 | 120K |
| ClaimExtraction | ~30,000 | 30 | 900K |
| CodeAnalysis | ~20,000 | 30 | 600K |
| GapAnalysis | ~8,000 | 30 | 240K |
| ContradictionMapping | ~2,000 × ~50 clusters | — | 100K |
| ReportGeneration | ~4,000 | 1 | 4K |
| **Total** | | | **~2.0M tokens** |

At Claude Sonnet 4.6 pricing (~$3/M input, ~$15/M output, 80/20 split): **$7–12 per 30-paper run**.

PDF text is cached to `artifacts/pdfs/` — re-runs save ~50% of ClaimExtraction and CodeAnalysis costs.

---

## Validation & Evaluation

ClaimCheck ships with a hand-annotation workflow for measuring agent accuracy against the research hypotheses.

### Validation set format (`tests/fixtures/validation_set.json`)

```json
[
  {
    "arxiv_id": "2106.09685",
    "ground_truth_claims": [
      {
        "metric": "accuracy", "dataset": "GLUE/MNLI",
        "model_base": "RoBERTa-large", "reported_value": 90.2,
        "conditions": {"rank": "8"}, "is_conditional": true
      }
    ],
    "known_gaps": [
      {
        "gap_type": "condition_undisclosed", "severity": "major",
        "description": "Paper uses rank=8 but default config sets rank=16"
      }
    ],
    "known_contradictions": []
  }
]
```

### Target metrics

| Metric | Target |
|---|---|
| Claim extraction precision | ≥ 0.85 |
| Claim extraction recall | ≥ 0.80 |
| Gap detection F1 | ≥ 0.75 |
| Contradiction precision | ≥ 0.70 |
| Pipeline coverage (no FAILED status) | ≥ 0.85 |

```bash
# Run validation evaluation
pytest tests/test_validation_set.py -v
```

---

## LoRA Variant Taxonomy (v1)

| Tag | Paper | ArXiv ID | Year |
|---|---|---|---|
| `LoRA` | LoRA: Low-Rank Adaptation of Large Language Models | 2106.09685 | 2021 |
| `QLoRA` | QLoRA: Efficient Finetuning of Quantized LLMs | 2305.14314 | 2023 |
| `AdaLoRA` | AdaLoRA: Adaptive Budget Allocation for PEFT | 2303.10512 | 2023 |
| `DoRA` | DoRA: Weight-Decomposed Low-Rank Adaptation | 2402.09353 | 2024 |
| `LoRA+` | LoRA+: Efficient Low Rank Adaptation of LLMs | 2402.12354 | 2024 |
| `VeRA` | VeRA: Vector-based Random Matrix Adaptation | 2310.11454 | 2023 |
| `DyLoRA` | DyLoRA: Parameter-Efficient Tuning with Dynamic Ranks | 2210.07558 | 2022 |
| `LoftQ` | LoftQ: LoRA-Fine-Tuning-Aware Quantization | 2310.08659 | 2023 |
| `LoRA-FA` | LoRA-FA: Memory-Efficient LLM Fine-Tuning | 2308.03303 | 2023 |
| `GLoRA` | One-for-All: Generalized LoRA for Parameter-Efficient Fine-Tuning | 2306.07967 | 2023 |
| `rsLoRA` | A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | 2312.03732 | 2023 |
| `MoLoRA` | Mixture of LoRA Experts | 2402.11453 | 2024 |
| `FLoRA` | Flora: Low-Rank Adapters Are Secretly Gradient Compressors | 2402.03293 | 2024 |

Papers not on this list are tagged `OTHER_LORA` and included but flagged for review.

---

## Deployment (Streamlit Community Cloud)

```bash
# 1. Push repo to GitHub (public)
git push origin main

# 2. Go to https://share.streamlit.io → "New app"
# 3. Select repo, branch: main, entrypoint: app/streamlit_app.py
# 4. Add secrets in "Advanced settings":
#      ANTHROPIC_API_KEY = "sk-ant-..."
#      GITHUB_TOKEN      = "ghp_..."
# 5. Click Deploy
```

The deployed app shows a **pre-computed demo dataset** (20 papers committed to `data/audit.db`). Full pipeline runs are available via the CLI locally.

---

## Limitations

| Limitation | Detail |
|---|---|
| Static analysis only | Code is read but not executed; silently-failing configs are indistinguishable from correct ones |
| PDF parsing quality | ~10–15% of arXiv PDFs require the `pymupdf` fallback; some scanned papers may still produce low-quality text |
| LLM accuracy ceiling | Unusual units or non-standard metric names may be miscategorised |
| Code availability | ~20–30% of LoRA papers have no released code; these reach `status=FAILED` at `code_analyzed` |
| Contradiction threshold | The 1% relative-difference threshold for `direct_numeric` is configurable but ultimately a design choice |
| v1 scope | Only LoRA-family PEFT methods; GPT-style prefix tuning, adapters, and prompt tuning are out of scope |

---

## Ethical Considerations

- **Fair representation** — Gap reports describe discrepancies in code/paper alignment, not author intent. No language implies fraud or misconduct.
- **Attribution** — Per-paper reports identify paper title + arXiv ID; individual authors are not named in negative contexts.
- **Uncertainty disclosure** — All LLM-generated findings are labelled as "automated analysis" and are hypotheses for human verification, not definitive conclusions.
- **Reproducibility of ClaimCheck itself** — Prompt templates, validation set, and evaluation scripts are committed so ClaimCheck's own methodology can be audited.
- **Data use** — Only publicly available arXiv papers and public GitHub repositories are accessed. No private repos, paywalled content, or personal data.

---

## Future Work

| Extension | Notes |
|---|---|
| Semantic Scholar integration | Add `tools/semantic_scholar_tool.py`; extend `PaperDiscoveryAgent` |
| Non-LoRA PEFT methods | Add `peft_family` field to `papers` table; parameterise agent prompts |
| Experiment re-execution | New agent that clones repo and runs training with detected config |
| Real-time pipeline progress | `st.empty()` + LangGraph streaming callbacks to push node events to UI |
| Multi-user deployment | Replace SQLite with PostgreSQL; add `user_id` to all tables |
| ArXiv alerting | Scheduled cron runs `PaperDiscoveryAgent` weekly; new papers auto-queued |
| CSV / JSON export | `st.download_button` on all Streamlit pages |

---

## Novel Academic Contributions

1. **A formal, reproducible methodology** for automated reproducibility auditing of PEFT literature — the first system to go beyond manual checking or paper summarization.
2. **An empirical dataset** — a structured corpus of 500+ benchmark claims with reproducibility labels and cross-paper contradiction tags across 20+ LoRA-variant papers, publishable as a standalone dataset contribution.
3. **A conditional claim schema** (`BenchmarkClaim.conditions` dict) that captures the hyperparameter context under which each claim is made — enabling comparison of "beats LoRA by 3%" claims that are only valid under specific rank/lr/dataset combinations.

---

*ClaimCheck — IISc Deep Learning Project · Built with LangGraph + Anthropic Claude + Streamlit*
