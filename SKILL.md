---
name: claimcheck-dev
description: >
  Development skill for the ClaimCheck LoRA Research Audit System.
  Guides GitHub Copilot when working on this codebase — generating agents,
  DB queries, Streamlit pages, LangGraph nodes, Pydantic models, and tests.
  Invoked when the user asks to: add an agent, create a DB query, add a chart,
  update the pipeline, write a test, or extend the Streamlit UI.
applyTo:
  - "**/*.py"
  - "db/**"
  - "app/**"
  - "agents/**"
  - "session/**"
  - "tools/**"
  - "tests/**"
---

# ClaimCheck Developer Skill

## Project Summary

ClaimCheck audits LoRA-variant ML research papers for:
1. **Reproducibility gaps** — paper-stated numbers vs. actual released code
2. **Cross-paper contradictions** — conflicting benchmark claims across papers

**Stack:** Python 3.11, LangGraph, Anthropic Claude, SQLite/aiosqlite, Pydantic v2, Streamlit, Plotly, NetworkX

---

## Code Generation Rules

### Every new Python file must start with:
```python
from __future__ import annotations
```

### Agent template
When asked to create or extend an agent, use this structure:

```python
# agents/{name}.py
from __future__ import annotations
import json
import uuid
from anthropic import AsyncAnthropic
from agents.base_agent import get_logger, run_with_limit, with_retry
from agents.llm import call_llm_json, make_client
from config import settings
from models import <InputModel>, <OutputModel>, PaperStatus
from db import queries

log = get_logger(__name__)

class {Name}Agent:
    async def run(self, input_model: <InputModel>) -> <OutputModel>:
        log_ctx = log.bind(paper_id=input_model.paper_id)
        try:
            # 1. Load data from DB
            # 2. Call tools / LLM
            # 3. Validate output
            # 4. Write to DB
            await queries.update_paper_status(input_model.paper_id, PaperStatus.{NEXT_STATUS})
            return <OutputModel>(...)
        except Exception as exc:
            log_ctx.error("agent_failed", exc=str(exc))
            await queries.update_paper_status(input_model.paper_id, PaperStatus.FAILED)
            raise
```

**PaperStatus transition table** (each agent advances to the next status on success):

| Agent | Input status | Output status on success |
|---|---|---|
| PaperDiscoveryAgent | — (creates record) | `DISCOVERED` |
| RepoResolutionAgent | `DISCOVERED` | `REPO_RESOLVED` |
| ClaimExtractionAgent | `REPO_RESOLVED` | `CLAIMS_EXTRACTED` |
| CodeAnalysisAgent | `CLAIMS_EXTRACTED` | `CODE_ANALYZED` |
| GapAnalysisAgent | `CODE_ANALYZED` | `GAPS_ANALYZED` |
| ContradictionMappingAgent | corpus-level (no per-paper status) | — |
| ReportGenerationAgent | corpus-level | sets all `GAPS_ANALYZED` → `DONE` |

### LLM call template
All LLM calls go through `agents/llm.py`. Never inline client construction or the JSON retry pattern.

```python
from agents.llm import call_llm_json, make_client

class MyAgent:
    def __init__(self, client=None):
        self.client = client or make_client()   # injectable for tests

    async def _call_llm(self, system: str, user: str) -> dict:
        # call_llm_json handles: client call, markdown fence stripping,
        # JSONDecodeError retry, and raises on second failure.
        return await call_llm_json(self.client, system, user)
```

### DB query template
When adding a query to `db/queries.py`:
```python
async def {operation}_{entity}({params}) -> {return_type}:
    """
    {Description}

    SQL:
        {SQL statement here}
    """
    now = _now()
    async with await _connect() as conn:
        await conn.execute(
            "SQL...",
            (param1, param2, now),
        )
        await conn.commit()
```

### Pydantic model template
When adding a model to `models.py`:
```python
class {Name}(BaseModel):
    {field}: {type}  # description
    # list[str] fields → stored as JSON TEXT in SQLite
    # dict fields → stored as JSON TEXT in SQLite
    # Optional fields default to None
```

### Streamlit page template
```python
# app/pages/{N}_{PageName}.py
import streamlit as st
import pandas as pd
from app.components.db_reader import get_{entity}_df
from app.components.charts import {chart_function}

st.header("{Page Title}")

@st.cache_data(ttl=30)
def load_data():
    return get_{entity}_df()

df = load_data()

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")
    # Add multiselect/checkbox filters

# Charts (always use plotly via charts.py)
st.plotly_chart({chart_function}(df), use_container_width=True)

# Data table
st.dataframe(df, use_container_width=True)
```

### Chart helper template
```python
# In app/components/charts.py
def {chart_name}(df: pd.DataFrame) -> go.Figure:
    """Description of what the chart shows."""
    # Always return go.Figure — never call st.plotly_chart here
    return px.{chart_type}(df, ...)
```

### LangGraph node template
```python
# In main.py
async def node_{name}(state: GraphState) -> GraphState:
    agent = {Name}Agent()
    result = await agent.run_all(state["paper_ids"])
    return {**state, "{result_key}": result}
```

### `run_user_session_from_text` template
Used by Streamlit pages — non-interactive version that bypasses stdin:
```python
# In session/user_session.py
async def run_user_session_from_text(
    user_text: str,
    variants: list[str] | str = "all",
) -> UserSessionOutput:
    """
    Non-interactive session for Streamlit UI.
    Runs LLM extraction only (no stdin, no confirmation).
    """
    data = await _call_llm_for_session(user_text)  # shared helper with run_user_session()
    if variants != "all" and variants != ["All"]:
        data["variants_of_interest"] = variants
    return UserSessionOutput(
        research_question=data["research_question"],
        variants_of_interest=data["variants_of_interest"],
        benchmarks_of_interest=data["benchmarks_of_interest"],
        search_queries=data["search_queries"] or ["LoRA fine-tuning"],
        raw_user_input=user_text,
        ambiguous=False,
    )
```

### Validation annotation helper template
For adding ground-truth claims to the validation set:
```python
# In tests/test_validation_set.py or a helper script
import json
from pathlib import Path
from models import BenchmarkClaim

def add_ground_truth_claim(arxiv_id: str, claim: dict, validation_path: Path) -> None:
    """
    Append a human-annotated ground-truth claim to validation_set.json.
    claim dict must match BenchmarkClaim schema (minus claim_id, paper_id).
    """
    vs = json.loads(validation_path.read_text()) if validation_path.exists() else []
    entry = next((e for e in vs if e["arxiv_id"] == arxiv_id), None)
    if entry is None:
        entry = {"arxiv_id": arxiv_id, "ground_truth_claims": [], "known_gaps": [], "known_contradictions": []}
        vs.append(entry)
    # Validate shape with Pydantic before appending
    _ = BenchmarkClaim(claim_id="validate-only", paper_id=arxiv_id, **claim)
    entry["ground_truth_claims"].append(claim)
    validation_path.write_text(json.dumps(vs, indent=2))
```

---

## Test generation rules

When generating tests, follow these patterns:

### Unit test (no live API):
```python
# tests/test_{module}.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_{function}_success():
    with patch("{module}.{external_call}", new_callable=AsyncMock) as mock:
        mock.return_value = {fixture_data}
        result = await {function}({inputs})
    assert result.{field} == {expected}

@pytest.mark.asyncio
async def test_{function}_handles_failure():
    with patch("{module}.{external_call}", side_effect=Exception("fail")):
        # assert graceful degradation, not exception propagation
```

### DB test:
```python
# tests/test_db.py
import pytest
import aiosqlite
from db import queries, init_db

@pytest.fixture
async def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    await init_db.create_tables(db_path)
    yield db_path

@pytest.mark.asyncio
async def test_insert_{entity}(test_db):
    entity = {EntityModel}(...)
    await queries.insert_{entity}(entity)
    results = await queries.get_{entity}_by_{field}(...)
    assert len(results) == 1
```

---

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Agent files | `snake_case.py` | `gap_analysis.py` |
| Agent classes | `PascalCaseAgent` | `GapAnalysisAgent` |
| DB query functions | `verb_noun()` | `insert_claim()`, `get_gaps_by_paper()` |
| Pydantic models | `PascalCase` | `BenchmarkClaim`, `ReproducibilityGap` |
| Streamlit pages | `NN_Title.py` | `03_Gaps.py` |
| Chart functions | `noun_chart()` or `noun_bar/donut/heatmap()` | `severity_donut()` |
| LangGraph nodes | `node_{name}` | `node_discover`, `node_gap_analysis` |
| Test files | `test_{module}.py` | `test_arxiv_tool.py` |

---

## Key File Locations

| Purpose | File |
|---|---|
| All Pydantic models | `models.py` |
| All DB queries | `db/queries.py` |
| DB schema | `db/schema.sql` |
| All settings | `config.py` |
| LangGraph graph | `main.py` → `build_graph()` |
| Shared agent utilities | `agents/base_agent.py` |
| Shared LLM helpers | `agents/llm.py` |
| Service layer (pipeline + MCP) | `services.py` |
| Streamlit chart helpers | `app/components/charts.py` |
| Streamlit DB readers | `app/components/db_reader.py` |
| PDF utilities | `tools/pdf_tool.py` |
| ArXiv utilities | `tools/arxiv_tool.py` |
| GitHub utilities | `tools/github_tool.py` |

---

## Do NOT

- Import agent code from Streamlit pages
- Call `asyncio.run()` inside async functions
- Construct SQL strings outside `db/queries.py`
- Call `pdf_tool.py` functions directly from async code (use `run_in_executor`)
- Use `print()` — always use `get_logger(__name__)`
- Hardcode API keys or paths — always use `settings`
- Add non-LoRA PEFT methods in v1 scope

---

## MCP Development Templates

Use these templates when the user asks for MCP support.

### MCP tool handler template
```python
# mcp_server/tools/{domain}.py
from __future__ import annotations
from mcp_server.schemas import {RequestModel}, {ResponseModel}
from mcp_server.auth import require_token
from services.{service_module} import {service_fn}

@require_token(mutable=True)
async def {tool_name}(payload: dict) -> dict:
    req = {RequestModel}.model_validate(payload)
    result = await {service_fn}(req)
    return {ResponseModel}.model_validate(result).model_dump()
```

### MCP run-tracking query template
```python
# db/queries.py
async def insert_pipeline_run(run_id: str, mode: str, trigger_source: str, status: str) -> None:
    """
    SQL:
        INSERT INTO pipeline_runs
        (run_id, mode, trigger_source, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """

async def update_pipeline_run_status(run_id: str, status: str, current_stage: str | None = None) -> None:
    """
    SQL:
        UPDATE pipeline_runs
        SET status = ?, current_stage = ?, updated_at = ?
        WHERE run_id = ?
    """
```

### MCP contract test template
```python
# tests/test_mcp_tools.py
import pytest

@pytest.mark.asyncio
async def test_start_pipeline_run_contract(mcp_client):
    resp = await mcp_client.call_tool(
        "start_pipeline_run",
        {"research_question": "QLoRA vs LoRA on MT-Bench", "variants": ["QLoRA"], "max_papers": 10},
    )
    assert "run_id" in resp
    assert resp["status"] in {"queued", "running"}

@pytest.mark.asyncio
async def test_mutating_tool_requires_auth(mcp_client_no_auth):
    with pytest.raises(Exception):
        await mcp_client_no_auth.call_tool("start_pipeline_run", {"research_question": "x"})
```

### MCP naming conventions

| Item | Convention | Example |
|---|---|---|
| MCP server package | `mcp_server/` | `mcp_server/server.py` |
| Tool handler files | `{domain}.py` | `mcp_server/tools/gaps.py` |
| Tool functions | `verb_noun` | `start_pipeline_run`, `get_run_status` |
| Request/response models | `{Tool}Request`/`{Tool}Response` | `StartPipelineRunRequest` |
| MCP tests | `test_mcp_*.py` | `test_mcp_tools.py` |
