# tests/test_db.py

from __future__ import annotations

import pytest

from db import queries
from models import Contradiction, PaperStatus, ReproducibilityGap


async def test_insert_and_get_paper(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    fetched = await queries.get_paper(sample_paper.arxiv_id)
    assert fetched is not None
    assert fetched.title == sample_paper.title
    assert fetched.authors == sample_paper.authors
    assert fetched.lora_variant_tag == "QLoRA"


async def test_insert_paper_idempotent(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    await queries.insert_paper(sample_paper)  # INSERT OR IGNORE
    papers = await queries.get_all_papers()
    assert len(papers) == 1


async def test_update_paper_status(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    await queries.update_paper_status(sample_paper.arxiv_id, PaperStatus.DONE)
    fetched = await queries.get_paper(sample_paper.arxiv_id)
    assert fetched.status == PaperStatus.DONE


async def test_update_paper_repo(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    await queries.update_paper_repo(
        sample_paper.arxiv_id, "https://github.com/x/y", 0.9
    )
    fetched = await queries.get_paper(sample_paper.arxiv_id)
    assert str(fetched.repo_url) == "https://github.com/x/y"
    assert fetched.repo_confidence == 0.9


async def test_get_papers_by_status(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    discovered = await queries.get_papers_by_status(PaperStatus.DISCOVERED)
    assert len(discovered) == 1
    done = await queries.get_papers_by_status(PaperStatus.DONE)
    assert done == []


async def test_claim_roundtrip(test_db, sample_paper, sample_claim):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    claims = await queries.get_claims_by_paper(sample_paper.arxiv_id)
    assert len(claims) == 1
    assert claims[0].conditions == {"rank": "8"}
    assert claims[0].is_conditional is True
    all_claims = await queries.get_all_claims()
    assert len(all_claims) == 1


async def test_code_fact_roundtrip(test_db, sample_paper, sample_fact):
    await queries.insert_paper(sample_paper)
    await queries.insert_code_fact(sample_fact)
    facts = await queries.get_facts_by_paper(sample_paper.arxiv_id)
    assert len(facts) == 1
    assert facts[0].line_range == (10, 12)
    assert facts[0].key == "rank"


async def test_gap_roundtrip(test_db, sample_paper, sample_claim):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    gap = ReproducibilityGap(
        gap_id="gap-1",
        paper_id=sample_paper.arxiv_id,
        claim_id=sample_claim.claim_id,
        gap_type="value_mismatch",
        severity="major",
        description="rank differs",
        paper_value="8",
        code_value="16",
    )
    await queries.insert_gap(gap)
    gaps = await queries.get_gaps_by_paper(sample_paper.arxiv_id)
    assert len(gaps) == 1
    assert gaps[0].severity == "major"
    all_gaps = await queries.get_all_gaps()
    assert len(all_gaps) == 1


async def test_contradiction_roundtrip(test_db, sample_paper):
    await queries.insert_paper(sample_paper)
    c = Contradiction(
        contradiction_id="c-1",
        paper_a_id="a",
        paper_b_id="b",
        claim_a_id="ca",
        claim_b_id="cb",
        contradiction_type="direct_numeric",
        description="values differ",
        severity="high",
    )
    await queries.insert_contradiction(c)
    contras = await queries.get_all_contradictions()
    assert len(contras) == 1
    assert contras[0].severity == "high"


async def test_audit_stats(test_db, sample_paper, sample_claim):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    stats = await queries.get_audit_stats()
    assert stats["papers_total"] == 1
    assert stats["claims_total"] == 1
    assert stats["claims_conditional"] == 1
    assert set(stats["papers_by_status"].keys()) == {s.value for s in PaperStatus}


async def test_pipeline_run_management(test_db):
    await queries.insert_pipeline_run("run-1", "question?", "all", 30)
    row = await queries.get_pipeline_run("run-1")
    assert row["status"] == "queued"
    await queries.update_pipeline_run(
        "run-1", status="running", current_stage="discover", progress=0.5
    )
    row = await queries.get_pipeline_run("run-1")
    assert row["status"] == "running"
    assert row["progress"] == 0.5
    assert row["current_stage"] == "discover"


async def test_mcp_tool_call_audit(test_db):
    await queries.insert_mcp_tool_call(
        "call-1", "run-1", "list_papers", "{}", "{}", "success", 12.5
    )
    # No exception means it persisted; verify via a direct connection
    conn = await queries._connect()
    try:
        cur = await conn.execute("SELECT COUNT(*) FROM mcp_tool_calls")
        count = (await cur.fetchone())[0]
    finally:
        await conn.close()
    assert count == 1
