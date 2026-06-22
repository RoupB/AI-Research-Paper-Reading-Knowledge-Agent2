# tests/test_pipeline_smoke.py
#
# End-to-end smoke test of the full LangGraph pipeline with mocked tools + LLM.
# Runs 2 papers through discover → report entirely offline.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from db import queries
from models import PaperStatus
from tools import arxiv_tool, github_tool

_AGENT_MODULES = [
    "agents.paper_discovery",
    "agents.repo_resolution",
    "agents.claim_extraction",
    "agents.code_analysis",
    "agents.gap_analysis",
    "agents.contradiction_mapping",
    "agents.report_generation",
]

_META = [
    {
        "arxiv_id": "2305.14314",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": ["Tim Dettmers"],
        "abstract": "We present QLoRA for efficient finetuning of quantized LLMs.",
        "published": "2023-05-23T00:00:00+00:00",
        "pdf_url": "https://arxiv.org/pdf/2305.14314",
        "arxiv_url": "https://arxiv.org/abs/2305.14314",
    },
    {
        "arxiv_id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward Hu"],
        "abstract": "We propose LoRA, low-rank adaptation of large language models.",
        "published": "2021-06-17T00:00:00+00:00",
        "pdf_url": "https://arxiv.org/pdf/2106.09685",
        "arxiv_url": "https://arxiv.org/abs/2106.09685",
    },
]


def _route(system: str) -> str:
    s = system.lower()
    if "research librarian" in s:
        return json.dumps({"relevant": True, "lora_variant_tag": "QLoRA", "reason": "yes"})
    if "official code repository" in s:
        return json.dumps({
            "github_urls_found": ["https://github.com/x/y"],
            "most_likely_url": "https://github.com/x/y",
            "confidence": 0.9, "reasoning": "found",
        })
    if "scientific claim extractor" in s:
        return json.dumps([{
            "metric": "accuracy", "dataset": "GLUE/MNLI", "model_base": "LLaMA-7B",
            "reported_value": 90.2, "unit": "%", "conditions": {"rank": "8"},
            "is_conditional": True, "claim_confidence": 0.9,
            "source_section": "Table 2", "raw_text": "90.2 on MNLI",
        }])
    if "code auditor" in s:
        return json.dumps([{
            "fact_type": "hyperparameter", "key": "rank", "value": "16",
            "file_path": "train.py", "line_range": [1, 2], "evidence": "rank=16",
        }])
    if "reproducibility auditor" in s:
        return json.dumps([{
            "gap_type": "value_mismatch", "severity": "major",
            "description": "rank 8 vs 16", "paper_value": "8", "code_value": "16",
            "claim_id": "__FIRST__", "fact_id": None,
        }])
    if "scientific fact-checker" in s:
        return json.dumps([])
    if "technical report writer" in s:
        return "Executive summary.\n\nKey Findings\n- finding"
    return json.dumps({})


def _make_routing_client() -> AsyncMock:
    client = AsyncMock()

    async def _create(*_args, **kwargs):
        system = kwargs.get("system", "")
        text = _route(system)
        msg = MagicMock()
        block = MagicMock()
        block.text = text
        msg.content = [block]
        return msg

    client.messages.create = AsyncMock(side_effect=_create)
    return client


@pytest.fixture
def _patch_world(monkeypatch, test_db):
    monkeypatch.setattr(settings, "discovery_min_papers", 1)

    async def fake_search(term, **kwargs):
        return _META

    async def fake_fetch_text(url, pages=None, full=False):
        return "Abstract\nWe report 90.2 accuracy on MNLI. https://github.com/x/y"

    async def fake_resolve(candidate, title, authors):
        return ("https://github.com/x/y", 0.9)

    async def fake_tree(repo_url, **kwargs):
        return [{"path": "train.py", "size": 50, "download_url": "u"}]

    async def fake_file(url):
        return "rank = 16"

    monkeypatch.setattr(arxiv_tool, "search_arxiv", fake_search)
    monkeypatch.setattr(arxiv_tool, "fetch_paper_text", fake_fetch_text)
    monkeypatch.setattr(github_tool, "resolve_repo", fake_resolve)
    monkeypatch.setattr(github_tool, "fetch_repo_tree", fake_tree)
    monkeypatch.setattr(github_tool, "fetch_file", fake_file)

    import importlib

    for mod_name in _AGENT_MODULES:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "make_client", _make_routing_client)


async def test_full_pipeline_smoke(_patch_world, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "report_output_dir", tmp_path / "reports")
    from main import build_graph

    graph = build_graph().compile()
    state = {
        "query_terms": ["QLoRA"],
        "variants_of_interest": "all",
        "benchmarks_of_interest": "all",
        "research_question": "Test",
        "paper_ids": [],
        "current_paper_id": None,
        "papers_processed": 0,
        "errors": [],
        "report_path": None,
    }
    result = await graph.ainvoke(state)

    # Papers discovered + persisted
    papers = await queries.get_all_papers()
    assert len(papers) == 2

    # Claims extracted
    claims = await queries.get_all_claims()
    assert len(claims) >= 2

    # Gaps analyzed
    gaps = await queries.get_all_gaps()
    assert len(gaps) >= 1

    # Report written
    report_path = result.get("report_path")
    assert report_path and Path(report_path).exists()

    # No paper stuck in a failed state
    stats = await queries.get_audit_stats()
    assert stats["papers_total"] == 2
