# tests/test_repo_resolution.py

from __future__ import annotations

import pytest

from agents.repo_resolution import RepoResolutionAgent
from db import queries
from models import PaperStatus, RepoResolutionInput
from tools import github_tool


async def test_resolves_and_persists(test_db, mock_llm, sample_paper, monkeypatch):
    await queries.insert_paper(sample_paper)

    async def fake_resolve(candidate, title, authors):
        return ("https://github.com/artidoro/qlora", 0.95)

    monkeypatch.setattr(github_tool, "resolve_repo", fake_resolve)

    client = mock_llm(
        {
            "github_urls_found": ["https://github.com/artidoro/qlora"],
            "most_likely_url": "https://github.com/artidoro/qlora",
            "confidence": 0.9,
            "reasoning": "found in pdf",
        }
    )
    agent = RepoResolutionAgent(client=client)
    inp = RepoResolutionInput(
        paper_id=sample_paper.arxiv_id,
        title=sample_paper.title,
        abstract=sample_paper.abstract,
        pdf_text_first_2_pages="See https://github.com/artidoro/qlora",
    )
    out = await agent.run(inp)
    assert out.repo_url == "https://github.com/artidoro/qlora"
    assert out.confidence == 0.95

    paper = await queries.get_paper(sample_paper.arxiv_id)
    assert paper.status == PaperStatus.REPO_RESOLVED
    assert str(paper.repo_url) == "https://github.com/artidoro/qlora"


async def test_low_confidence_nulls_repo(test_db, mock_llm, sample_paper, monkeypatch):
    await queries.insert_paper(sample_paper)

    async def fake_resolve(candidate, title, authors):
        return ("https://github.com/maybe/repo", 0.3)

    monkeypatch.setattr(github_tool, "resolve_repo", fake_resolve)
    client = mock_llm(
        {"github_urls_found": [], "most_likely_url": None, "confidence": 0.3, "reasoning": "guess"}
    )
    agent = RepoResolutionAgent(client=client)
    inp = RepoResolutionInput(
        paper_id=sample_paper.arxiv_id,
        title=sample_paper.title,
        abstract=sample_paper.abstract,
        pdf_text_first_2_pages="no links here",
    )
    out = await agent.run(inp)
    assert out.repo_url is None
    paper = await queries.get_paper(sample_paper.arxiv_id)
    assert paper.repo_url is None
