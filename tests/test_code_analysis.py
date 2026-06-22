# tests/test_code_analysis.py

from __future__ import annotations

import json

import pytest

from agents.code_analysis import CodeAnalysisAgent
from db import queries
from models import CodeAnalysisInput, PaperStatus
from tools import github_tool

_FACTS = [
    {
        "fact_type": "hyperparameter",
        "key": "rank",
        "value": "16",
        "file_path": "train.py",
        "line_range": [10, 11],
        "evidence": "parser.add_argument('--rank', default=16)",
    },
    {
        "fact_type": "dataset",
        "key": "MNLI",
        "value": "glue/mnli",
        "file_path": "train.py",
        "line_range": None,
        "evidence": "load_dataset('glue', 'mnli')",
    },
]


async def test_code_analysis_persists_facts(test_db, mock_llm, sample_paper, monkeypatch):
    await queries.insert_paper(sample_paper)

    async def fake_tree(repo_url, **kwargs):
        return [
            {"path": "train.py", "size": 100,
             "download_url": "https://raw.githubusercontent.com/x/y/main/train.py"}
        ]

    async def fake_file(url):
        return "parser.add_argument('--rank', default=16)\nload_dataset('glue','mnli')"

    monkeypatch.setattr(github_tool, "fetch_repo_tree", fake_tree)
    monkeypatch.setattr(github_tool, "fetch_file", fake_file)

    client = mock_llm(json.dumps(_FACTS))
    agent = CodeAnalysisAgent(client=client)
    inp = CodeAnalysisInput(paper_id=sample_paper.arxiv_id, repo_url="https://github.com/x/y")
    out = await agent.run(inp)
    assert out.facts_extracted == 2
    facts = await queries.get_facts_by_paper(sample_paper.arxiv_id)
    keys = {f.key for f in facts}
    assert "rank" in keys
    paper = await queries.get_paper(sample_paper.arxiv_id)
    assert paper.status == PaperStatus.CODE_ANALYZED


async def test_no_python_files_writes_missing_eval(test_db, mock_llm, sample_paper, monkeypatch):
    await queries.insert_paper(sample_paper)

    async def fake_tree(repo_url, **kwargs):
        return [{"path": "README.md", "size": 10, "download_url": "u"}]

    monkeypatch.setattr(github_tool, "fetch_repo_tree", fake_tree)
    client = mock_llm("[]")
    agent = CodeAnalysisAgent(client=client)
    inp = CodeAnalysisInput(paper_id=sample_paper.arxiv_id, repo_url="https://github.com/x/y")
    out = await agent.run(inp)
    assert out.facts_extracted == 1
    facts = await queries.get_facts_by_paper(sample_paper.arxiv_id)
    assert facts[0].fact_type == "missing_eval"
    assert facts[0].key == "no_python_code"


def test_chunks_large_file():
    content = "\n".join(f"line {i}" for i in range(900))
    chunks = CodeAnalysisAgent._chunks(content)
    assert len(chunks) > 1
