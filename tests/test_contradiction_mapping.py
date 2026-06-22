# tests/test_contradiction_mapping.py

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.contradiction_mapping import ContradictionMappingAgent
from db import queries
from models import BenchmarkClaim, Paper, PaperStatus


def _paper(aid: str, tag: str) -> Paper:
    return Paper(
        arxiv_id=aid,
        title=f"Paper {aid}",
        authors=["A"],
        abstract="abstract",
        published=datetime(2023, 1, 1, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/{aid}",
        arxiv_url=f"https://arxiv.org/abs/{aid}",
        lora_variant_tag=tag,
        status=PaperStatus.DONE,
    )


def _claim(paper_id: str, cid: str, value: float) -> BenchmarkClaim:
    return BenchmarkClaim(
        paper_id=paper_id,
        claim_id=cid,
        metric="accuracy",
        dataset="GLUE/MNLI",
        model_base="LLaMA-7B",
        reported_value=value,
        unit="%",
        source_section="Table 1",
        raw_text="text",
    )


async def test_contradiction_detected(test_db, mock_llm):
    await queries.insert_paper(_paper("2305.14314", "QLoRA"))
    await queries.insert_paper(_paper("2106.09685", "LoRA"))
    await queries.insert_claim(_claim("2305.14314", "c1", 90.2))
    await queries.insert_claim(_claim("2106.09685", "c2", 85.0))

    resp = [
        {
            "paper_a_id": "2305.14314",
            "paper_b_id": "2106.09685",
            "claim_a_id": "c1",
            "claim_b_id": "c2",
            "contradiction_type": "direct_numeric",
            "description": "90.2 vs 85.0 on same setup.",
            "severity": "high",
        }
    ]
    client = mock_llm(json.dumps(resp))
    agent = ContradictionMappingAgent(client=client)
    out = await agent.run()
    assert out.contradictions_found == 1
    contras = await queries.get_all_contradictions()
    assert contras[0].contradiction_type == "direct_numeric"
    assert set(out.papers_involved) == {"2305.14314", "2106.09685"}


async def test_single_paper_cluster_skipped(test_db, mock_llm):
    await queries.insert_paper(_paper("2305.14314", "QLoRA"))
    await queries.insert_claim(_claim("2305.14314", "c1", 90.2))
    await queries.insert_claim(_claim("2305.14314", "c2", 91.0))
    client = mock_llm("[]")  # should never be called
    agent = ContradictionMappingAgent(client=client)
    out = await agent.run()
    assert out.contradictions_found == 0


async def test_empty_corpus(test_db, mock_llm):
    client = mock_llm("[]")
    agent = ContradictionMappingAgent(client=client)
    out = await agent.run()
    assert out.contradictions_found == 0
    assert out.papers_involved == []


async def test_hallucinated_claim_ids_discarded(test_db, mock_llm):
    await queries.insert_paper(_paper("2305.14314", "QLoRA"))
    await queries.insert_paper(_paper("2106.09685", "LoRA"))
    await queries.insert_claim(_claim("2305.14314", "c1", 90.2))
    await queries.insert_claim(_claim("2106.09685", "c2", 85.0))
    resp = [
        {
            "paper_a_id": "2305.14314",
            "paper_b_id": "2106.09685",
            "claim_a_id": "nonexistent",
            "claim_b_id": "c2",
            "contradiction_type": "direct_numeric",
            "description": "bad ids",
            "severity": "high",
        }
    ]
    client = mock_llm(json.dumps(resp))
    agent = ContradictionMappingAgent(client=client)
    out = await agent.run()
    assert out.contradictions_found == 0
