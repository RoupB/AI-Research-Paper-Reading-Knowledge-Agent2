# tests/test_gap_analysis.py

from __future__ import annotations

import json

import pytest

from agents.gap_analysis import GapAnalysisAgent
from db import queries
from models import GapAnalysisInput, PaperStatus


async def test_gap_from_claims_and_facts(test_db, mock_llm, sample_paper, sample_claim, sample_fact):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)

    gap_resp = [
        {
            "gap_type": "value_mismatch",
            "severity": "major",
            "description": "Paper uses rank=8 but code default is 16.",
            "paper_value": "8",
            "code_value": "16",
            "claim_id": sample_claim.claim_id,
            "fact_id": sample_fact.fact_id,
        }
    ]
    client = mock_llm(json.dumps(gap_resp))
    agent = GapAnalysisAgent(client=client)
    inp = GapAnalysisInput(
        paper_id=sample_paper.arxiv_id,
        claims=[sample_claim],
        code_facts=[sample_fact],
    )
    out = await agent.run(inp)
    assert out.gaps_found == 1
    assert out.severity_counts["major"] == 1
    gaps = await queries.get_gaps_by_paper(sample_paper.arxiv_id)
    assert gaps[0].gap_type == "value_mismatch"


async def test_empty_inputs_create_critical_gap(test_db, mock_llm, sample_paper, sample_claim):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    client = mock_llm("[]")
    agent = GapAnalysisAgent(client=client)
    inp = GapAnalysisInput(
        paper_id=sample_paper.arxiv_id, claims=[sample_claim], code_facts=[]
    )
    out = await agent.run(inp)
    assert out.gaps_found == 1
    gaps = await queries.get_gaps_by_paper(sample_paper.arxiv_id)
    assert gaps[0].severity == "critical"
    assert gaps[0].gap_type == "missing_code"


async def test_invalid_claim_id_discarded(test_db, mock_llm, sample_paper, sample_claim, sample_fact):
    await queries.insert_paper(sample_paper)
    await queries.insert_claim(sample_claim)
    gap_resp = [
        {
            "gap_type": "missing_code",
            "severity": "weird_severity",  # coerced to minor
            "description": "x",
            "claim_id": "hallucinated-id",  # invalid → remapped
            "fact_id": None,
        }
    ]
    client = mock_llm(json.dumps(gap_resp))
    agent = GapAnalysisAgent(client=client)
    inp = GapAnalysisInput(
        paper_id=sample_paper.arxiv_id, claims=[sample_claim], code_facts=[sample_fact]
    )
    out = await agent.run(inp)
    assert out.gaps_found == 1
    gaps = await queries.get_gaps_by_paper(sample_paper.arxiv_id)
    assert gaps[0].severity == "minor"
    assert gaps[0].claim_id == sample_claim.claim_id
