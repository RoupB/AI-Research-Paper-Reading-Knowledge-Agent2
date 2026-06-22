# tests/test_mcp.py

from __future__ import annotations

import pytest
from pydantic import ValidationError

import services
from config import settings
from db import queries
from mcp_server import auth, server


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    auth.reset_rate_limits()
    yield
    auth.reset_rate_limits()


async def test_dispatch_unknown_tool():
    with pytest.raises(KeyError):
        await server.dispatch("not_a_tool", {})


async def test_list_papers_tool(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    result = await server.dispatch("list_papers", {})
    assert result["count"] == 1
    assert result["papers"][0]["arxiv_id"] == sample_paper.arxiv_id


async def test_get_claims_tool(test_db, sample_paper, sample_claim):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    result = await server.dispatch("get_claims", {"paper_id": sample_paper.arxiv_id})
    assert result["count"] == 1


async def test_get_code_facts_requires_paper_id(test_db):
    with pytest.raises(ValidationError):
        await server.dispatch("get_code_facts", {})


async def test_auth_required_for_mutating_tool(test_db, monkeypatch):
    async def fake_report(**kwargs):
        return {
            "report_md_path": "a.md", "report_html_path": "a.html",
            "papers_in_report": 0, "total_gaps": 0, "total_contradictions": 0,
        }

    monkeypatch.setattr(services, "generate_report", fake_report)

    # No token → AuthError
    with pytest.raises(auth.AuthError):
        await server.dispatch(
            "generate_report", {"run_id": "r1"}, token=None
        )

    # Correct token → succeeds
    result = await server.dispatch(
        "generate_report", {"run_id": "r1"}, token=settings.mcp_auth_token
    )
    assert result["report_md_path"] == "a.md"


async def test_rate_limit(test_db, sample_paper, monkeypatch):
    await queries.insert_paper(sample_paper)
    monkeypatch.setattr(settings, "mcp_rate_limit_per_min", 3)
    auth.reset_rate_limits()
    for _ in range(3):
        await server.dispatch("list_papers", {})
    with pytest.raises(auth.RateLimitError):
        await server.dispatch("list_papers", {})


async def test_audit_record_persisted(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    await server.dispatch("list_papers", {})
    conn = await queries._connect()
    try:
        cur = await conn.execute(
            "SELECT tool_name, status FROM mcp_tool_calls WHERE tool_name='list_papers'"
        )
        row = await cur.fetchone()
    finally:
        await conn.close()
    assert row is not None
    assert row[1] == "success"


async def test_start_run_requires_auth(test_db):
    with pytest.raises(auth.AuthError):
        await server.dispatch(
            "start_pipeline_run", {"research_question": "q"}, token="wrong"
        )


def test_tool_registry_complete():
    expected = {
        "start_pipeline_run", "get_run_status", "list_papers", "get_claims",
        "get_code_facts", "get_gaps", "get_contradictions", "generate_report",
    }
    assert set(server.TOOL_REGISTRY.keys()) == expected
