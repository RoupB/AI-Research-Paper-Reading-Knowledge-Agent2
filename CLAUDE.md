# CLAUDE.md — Agent Instructions for ClaimCheck

This file is read by Claude (and compatible AI agents) when working inside this repository.
Follow these instructions precisely when assisting with code generation, debugging, or analysis.

---

## Project Identity

**ClaimCheck** is an agentic pipeline that audits ML research papers (LoRA-variant fine-tuning methods) for:
1. Reproducibility gaps — does the released code match paper claims?
2. Cross-paper contradictions — do different papers' benchmark numbers conflict?

The system is a **LangGraph**-based multi-agent pipeline with a **Streamlit** frontend, backed by **SQLite**, and deployed on **Streamlit Community Cloud**.

---

## Architecture Overview

```
session/user_session.py          ← Interactive CLI scoping (runs before pipeline)
agents/paper_discovery.py        ← Agent 1: ArXiv search
agents/repo_resolution.py        ← Agent 2: GitHub repo finder
agents/claim_extraction.py       ← Agent 3: PDF → structured benchmark claims
agents/code_analysis.py          ← Agent 4: GitHub repo → code facts
agents/gap_analysis.py           ← Agent 5: claims vs code → gaps
agents/contradiction_mapping.py  ← Agent 6: cross-paper contradictions
agents/report_generation.py      ← Agent 7: Markdown + HTML report
app/streamlit_app.py             ← Streamlit frontend
main.py                          ← LangGraph graph + typer CLI
```

All agents share a single **SQLite** database (`data/audit.db`). No agent writes to another agent's tables. The pipeline status of each paper advances through:
`discovered → repo_resolved → claims_extracted → code_analyzed → gaps_analyzed → done`

---

## Coding Conventions

### Language & Runtime
- Python 3.11 only
- All agent `run()` methods are `async`. Use `await` throughout — never `asyncio.run()` inside agent code (only in `main.py` entrypoint)
- `pdf_tool.py` functions are **synchronous** — always call them via `asyncio.run_in_executor(None, fn, *args)`

### Imports
```python
from __future__ import annotations   # always first in every file
```

- Never import `settings` inside function bodies — import at module level
- Import order: stdlib → third-party → local (config, models, db, tools, agents)

### Pydantic
- All agent inputs/outputs are Pydantic v2 `BaseModel`
- Validate input at entry: `model = InputModel.model_validate(data)`
- Serialize output before DB write: `model.model_dump()`
- Never construct dicts manually when a model exists

### LLM Calls
- Always use the `anthropic` SDK directly (not LangChain's wrapper)
- Every LLM call must expect **JSON output only**
- JSON retry pattern (mandatory — do not deviate):
  ```python
  response = await call_llm(prompt)
  try:
      data = json.loads(response)
  except json.JSONDecodeError:
      response = await call_llm("Respond only with JSON, no prose.\n" + prompt)
      data = json.loads(response)   # raises on second failure → caught by caller
  ```
- Temperature: `settings.anthropic_temperature` (default 0.1) for all extraction tasks
- **Cost awareness**: Before adding a new LLM call, estimate its token cost. Full-paper calls (ClaimExtraction, CodeAnalysis) are expensive (~30K tokens each). Prefer structured JSON inputs over raw text re-sending. Never re-send the full paper text to a downstream agent — use the structured DB records instead.

### Database
- All DB operations go through `db/queries.py` — no agent constructs SQL directly
- Every `queries.py` function is `async` and opens its own `aiosqlite` connection
- `PRAGMA journal_mode=WAL` is set on every connection
- `created_at` / `updated_at` are set by `queries.py`, not by callers

### Logging
```python
from agents.base_agent import get_logger
log = get_logger(__name__)
log = log.bind(paper_id=paper_id)  # bind context before sub-operations
```
- Log levels: `DEBUG` (trace), `INFO` (normal events), `WARNING` (recoverable, pipeline continues), `ERROR` (paper marked FAILED)
- Never use `print()` — use structlog

### Error Handling
- If a paper fails at any stage: set `status=FAILED`, log `ERROR`, **continue** the pipeline for other papers
- Never raise from within an agent's `run()` except for unrecoverable config errors
- All HTTP calls use `with_retry()` from `base_agent.py`

---

## Agentic Workflow

When implementing a new agent or modifying an existing one, follow this workflow:

### Agentic Discovery Loop (PaperDiscoveryAgent)

`PaperDiscoveryAgent` is the **only** agent that runs as a refinement loop, not a single-pass operation. The loop runs up to `DISCOVERY_MAX_ROUNDS` (default 3) times:

```
Round 1: Search with all query_terms from UserSessionOutput
  → If papers found < MIN_PAPERS_THRESHOLD (default 10):
Round 2: Widen queries (add "survey", "benchmark", strip variant names)
  → If still < threshold:
Round 3: Fall back to broad query "parameter efficient fine-tuning LoRA"
  → Accept whatever is found; log WARNING "discovery_below_threshold"
```

After each round, filter results through the LLM relevance check. The loop exits early if the threshold is met. This is what the README calls an "agentic loop that refines search" — it is explicit iteration, not a free-form ReAct loop.

All other agents (2–7) are **single-pass**: they run exactly once per paper or once per corpus.

---

### Step 1 — Read the spec
Always read `spec.md` before touching agent code. The spec defines exact input/output models, LLM prompts, error handling rules, and DB interactions.

### Step 2 — Validate models first
Define or verify Pydantic `Input` and `Output` models in `models.py` before writing agent logic.

### Step 3 — Write the DB layer first
If an agent needs new DB reads/writes, add them to `db/queries.py` first with docstrings and SQL comments. Write `tests/test_db.py` tests before the agent itself.

### Step 4 — Implement tools independently
`tools/` functions have no agent dependencies. Implement and test them in isolation (`tests/test_*_tool.py`) before agents use them.

### Step 5 — Implement the agent
Follow the build order in `spec.md`. An agent should:
1. Accept its Pydantic input model
2. Load required DB records via `queries.py`
3. Call tools (via `run_with_limit` if concurrent)
4. Call LLM with the exact prompt from spec
5. Validate LLM output against Pydantic model
6. Write results to DB via `queries.py`
7. Update paper status via `update_paper_status()`
8. Return the Pydantic output model

### Step 6 — Wire into LangGraph
Add the agent as a node in `main.py`'s `build_graph()`. Add conditional edges if the agent can be skipped (e.g., no repo → skip code analysis).

### Step 7 — Add Streamlit UI
If the agent produces data that should be displayed, add/update the relevant page in `app/pages/`.

---

## LangGraph-Specific Rules

- State type is `GraphState` (TypedDict) in `main.py`
- Each node function signature: `async def node_X(state: GraphState) -> GraphState`
- Nodes return a **new state dict** (`{**state, "key": new_value}`) — never mutate in place
- Use `SqliteSaver` checkpointer so `--resume` restarts from last completed node
- Conditional edges: return a string key that maps to the next node name or `END`

---

## Streamlit Rules

- Streamlit pages are in `app/pages/` using multi-page naming (`01_Search.py`, etc.)
- All DB reads from Streamlit use `app/components/db_reader.py` (sync SQLite, not aiosqlite)
- All charts use `app/components/charts.py` — return `go.Figure` objects, never render inside helpers
- Use `st.cache_data` on expensive DB reads:
  ```python
  @st.cache_data(ttl=30)
  def get_gaps_df() -> pd.DataFrame: ...
  ```
- Never import agent or pipeline code from Streamlit pages — only `db_reader` and `charts`
- Secrets are loaded from Streamlit Cloud's secrets UI (maps to `.env` vars)

---

## File Modification Rules

| File | Rule |
|---|---|
| `models.py` | Add new models here; never define models inline in agent files |
| `db/schema.sql` | Add columns/tables here; run `db/init_db.py` to rebuild |
| `db/queries.py` | One function per SQL operation; always document with SQL comment |
| `agents/base_agent.py` | Do not add agent-specific logic; only shared utilities |
| `config.py` | Add new settings as typed fields with defaults; never hardcode values |
| `main.py` | Only pipeline wiring (LangGraph nodes/edges) + CLI commands |
| `app/components/charts.py` | Chart helpers only — no DB calls, no Streamlit widgets |
| `app/components/db_reader.py` | Read-only; never write to DB from here |

---

## What NOT to Do

- Do not run training or re-run ML experiments
- Do not open unsolicited GitHub PRs or post paper rebuttals
- Do not add non-LoRA PEFT methods in v1
- Do not use `print()` — use `structlog`
- Do not construct SQL strings in agent files
- Do not call `asyncio.run()` inside async code
- Do not call `pdf_tool.py` functions directly from async agents (use `run_in_executor`)
- Do not skip input validation on agent entry

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| `pdf_tool` functions blocking event loop | Wrap in `await loop.run_in_executor(None, fn, *args)` |
| LLM returning extra prose around JSON | Always use the JSON retry pattern |
| GitHub 403 on repo fetch | Check `GITHUB_TOKEN` is set; use `repo_confidence < 0.5` → skip |
| SQLite "database is locked" | Ensure `PRAGMA journal_mode=WAL` is set on every connection |
| Streamlit `st.cache_data` stale after pipeline run | Set `ttl=30` and call `st.cache_data.clear()` after pipeline completes |
| LangGraph node returning wrong keys | Always spread existing state: `{**state, "new_key": val}` |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python db/init_db.py

# Run interactive pipeline
python main.py run --max 30

# Run non-interactive (for scripting/testing)
python main.py run --max 30 --resume

# Launch Streamlit UI
python main.py ui
# OR directly:
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# Run with debug logging
LOG_LEVEL=DEBUG python main.py run --max 5
```

---

## session/user_session.py — Two Entry Points

| Function | Used by | Behaviour |
|---|---|---|
| `run_user_session()` | `main.py` CLI | Full interactive: stdin prompts, clarification, confirmation |
| `run_user_session_from_text(user_text, variants)` | `app/pages/01_Search.py` | Non-interactive: receives pre-typed text from Streamlit form, runs LLM only |

Both return `UserSessionOutput`. The Streamlit page always uses `run_user_session_from_text()` — never `run_user_session()` (which would block on stdin).

---

## Environment Variables (required)

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) |
| `GITHUB_TOKEN` | GitHub Personal Access Token, `read:public_repo` scope (required) |

See `.env.example` for the full list with defaults.

---

## MCP Architecture Rules

ClaimCheck supports an MCP deployment mode. In MCP mode, the system exposes tool endpoints but reuses the same core pipeline logic.

### Architectural constraints
- Do not duplicate agent business logic in MCP tool handlers.
- MCP handlers must call shared service functions (service-first architecture).
- Keep LangGraph orchestration as the canonical control flow for full pipeline runs.
- MCP tooling is an interface layer, not an alternative implementation.

### Required MCP modules
```
mcp_server/
├── server.py
├── schemas.py
├── auth.py
├── run_manager.py
└── tools/
  ├── pipeline.py
  ├── papers.py
  ├── claims.py
  ├── code.py
  ├── gaps.py
  ├── contradictions.py
  └── report.py
```

### MCP run management
- Every mutating MCP call must include `run_id`.
- Persist run state in `pipeline_runs`.
- Persist tool-call audit records in `mcp_tool_calls`.
- Include latency and status (`success`/`error`) for each tool call.

### MCP security
- Validate `MCP_AUTH_TOKEN` for mutating tools.
- Apply per-tool rate limits.
- Never return secrets in tool responses.
- Log all request/response payloads in redacted form.

### MCP testing requirements
- Contract tests for each tool (input/output schema validation).
- Integration test: start run -> poll status -> generate report.
- Negative tests for auth failure, invalid input, timeout, and cancellation.
