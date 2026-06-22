# tests/test_claim_extraction.py

from __future__ import annotations

import json

import pytest

from agents.claim_extraction import ClaimExtractionAgent
from db import queries
from models import ClaimExtractionInput, PaperStatus

_CLAIMS = [
    {
        "metric": "accuracy",
        "dataset": "GLUE/MNLI",
        "model_base": "LLaMA-7B",
        "reported_value": 90.2,
        "unit": "%",
        "conditions": {"rank": "8"},
        "is_conditional": True,
        "claim_confidence": 0.9,
        "source_section": "Table 2",
        "raw_text": "QLoRA achieves 90.2% on MNLI.",
    },
    {
        "metric": "accuracy",
        "dataset": "GLUE/MNLI",
        "model_base": "LLaMA-7B",
        "reported_value": 90.2,  # duplicate → deduped
        "unit": "%",
        "conditions": {},
        "is_conditional": False,
        "claim_confidence": 0.8,
        "source_section": "Section 4",
        "raw_text": "duplicate",
    },
    {
        "metric": "perplexity",
        "dataset": "WikiText",
        "model_base": "LLaMA-7B",
        "reported_value": 5.6,
        "unit": None,
        "conditions": {},
        "is_conditional": False,
        "claim_confidence": 0.95,
        "source_section": "Table 3",
        "raw_text": "Perplexity of 5.6.",
    },
]


async def test_extraction_dedups_and_persists(test_db, mock_llm, sample_paper):
    await queries.insert_paper(sample_paper)
    client = mock_llm(json.dumps(_CLAIMS))  # one JSON-array response
    agent = ClaimExtractionAgent(client=client)
    inp = ClaimExtractionInput(
        paper_id=sample_paper.arxiv_id,
        pdf_url=str(sample_paper.pdf_url),
        full_text="Some paper text with results.",
    )
    out = await agent.run(inp)
    assert out.claims_extracted == 2  # one duplicate removed
    claims = await queries.get_claims_by_paper(sample_paper.arxiv_id)
    assert len(claims) == 2
    paper = await queries.get_paper(sample_paper.arxiv_id)
    assert paper.status == PaperStatus.CLAIMS_EXTRACTED


async def test_no_claims_marks_failed(test_db, mock_llm, sample_paper):
    await queries.insert_paper(sample_paper)
    client = mock_llm("[]")  # empty JSON array
    agent = ClaimExtractionAgent(client=client)
    inp = ClaimExtractionInput(
        paper_id=sample_paper.arxiv_id,
        pdf_url=str(sample_paper.pdf_url),
        full_text="A survey paper with no numbers.",
    )
    out = await agent.run(inp)
    assert out.claims_extracted == 0
    paper = await queries.get_paper(sample_paper.arxiv_id)
    assert paper.status == PaperStatus.FAILED


def test_windows_splits_large_text():
    big = "x" * 80_000
    windows = ClaimExtractionAgent._windows(big)
    assert len(windows) > 1
