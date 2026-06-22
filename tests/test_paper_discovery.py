# tests/test_paper_discovery.py

from __future__ import annotations

import pytest

from agents.paper_discovery import PaperDiscoveryAgent
from config import settings
from db import queries
from tools import arxiv_tool

_META = [
    {
        "arxiv_id": "2305.14314",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": ["Tim Dettmers"],
        "abstract": "We present QLoRA, efficient finetuning of quantized LLMs.",
        "published": "2023-05-23T00:00:00+00:00",
        "pdf_url": "https://arxiv.org/pdf/2305.14314",
        "arxiv_url": "https://arxiv.org/abs/2305.14314",
    },
    {
        "arxiv_id": "9999.00000",
        "title": "Unrelated Vision Paper",
        "authors": ["Someone"],
        "abstract": "A paper about convolutional networks for images.",
        "published": "2022-01-01T00:00:00+00:00",
        "pdf_url": "https://arxiv.org/pdf/9999.00000",
        "arxiv_url": "https://arxiv.org/abs/9999.00000",
    },
]


async def test_discovery_filters_and_persists(test_db, mock_llm, monkeypatch):
    monkeypatch.setattr(settings, "discovery_min_papers", 1)

    async def fake_search(term, **kwargs):
        return _META

    monkeypatch.setattr(arxiv_tool, "search_arxiv", fake_search)

    client = mock_llm(
        [
            {"relevant": True, "lora_variant_tag": "QLoRA", "reason": "yes"},
            {"relevant": False, "lora_variant_tag": None, "reason": "vision"},
        ]
    )
    agent = PaperDiscoveryAgent(client=client)
    out = await agent.run(["QLoRA"])

    assert out.papers_found == 1
    assert "2305.14314" in out.paper_ids
    papers = await queries.get_all_papers()
    assert len(papers) == 1
    assert papers[0].lora_variant_tag == "QLoRA"


async def test_unknown_variant_tagged_other(test_db, mock_llm, monkeypatch):
    monkeypatch.setattr(settings, "discovery_min_papers", 1)

    async def fake_search(term, **kwargs):
        return [_META[0]]

    monkeypatch.setattr(arxiv_tool, "search_arxiv", fake_search)
    client = mock_llm({"relevant": True, "lora_variant_tag": "SuperLoRA", "reason": "novel"})
    agent = PaperDiscoveryAgent(client=client)
    out = await agent.run(["LoRA"])
    assert out.papers_found == 1
    papers = await queries.get_all_papers()
    assert papers[0].lora_variant_tag == "OTHER_LORA"


def test_normalise_tag():
    assert PaperDiscoveryAgent._normalise_tag("QLoRA") == "QLoRA"
    assert PaperDiscoveryAgent._normalise_tag("qlora") == "qlora"
    assert PaperDiscoveryAgent._normalise_tag("WeirdThing") == "OTHER_LORA"
    assert PaperDiscoveryAgent._normalise_tag(None) == "OTHER_LORA"
