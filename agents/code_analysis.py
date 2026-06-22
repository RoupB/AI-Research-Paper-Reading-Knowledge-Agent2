# agents/code_analysis.py

from __future__ import annotations
import asyncio
import uuid

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger, run_with_limit
from agents.llm import call_llm_json, make_client
from db import queries
from models import (
    CodeAnalysisInput,
    CodeAnalysisOutput,
    CodeFact,
    PaperStatus,
)
from tools import github_tool

log = get_logger(__name__)

_SYSTEM = """You are a code auditor analyzing ML training scripts for reproducibility.
Given source code from a research repository, extract concrete facts about:
1. Hyperparameters: any hardcoded or argparse-default values (rank, alpha, lr,
   batch_size, epochs, warmup_steps, dropout, etc.)
2. Datasets: dataset names, loading paths, splits used
3. Metrics logged: what metrics are actually computed and saved
4. Evaluation coverage: which benchmark datasets have eval scripts present

For each fact return:
- fact_type: "hyperparameter" | "dataset" | "metric_logged" | "missing_eval"
- key: the parameter/dataset/metric name
- value: the value found in code (or null if missing_eval)
- file_path: relative path within the repo
- line_range: [start, end] if identifiable
- evidence: the exact code snippet (max 5 lines)

Return a JSON array of fact objects."""

_MAX_FILES = 12
_CHUNK_LINES = 200
_CHUNK_OVERLAP = 20


class CodeAnalysisAgent:
    """Agent 4 — extracts reproducibility-relevant facts from a repository."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def _analyze_file(
        self, repo_url: str, paper_id: str, file_path: str, content: str
    ) -> list[dict]:
        user = (
            f"Repository: {repo_url}\nPaper ID: {paper_id}\n\n"
            f"File: {file_path}\n```\n{content}\n```\n\n"
            "Extract all reproducibility-relevant facts from this file."
        )
        data = await call_llm_json(self.client, _SYSTEM, user)
        if isinstance(data, dict):
            data = data.get("facts", [])
        return data if isinstance(data, list) else []

    @staticmethod
    def _chunks(content: str) -> list[str]:
        lines = content.splitlines()
        if len(lines) <= 500:
            return [content]
        out: list[str] = []
        start = 0
        while start < len(lines):
            out.append("\n".join(lines[start : start + _CHUNK_LINES]))
            start += _CHUNK_LINES - _CHUNK_OVERLAP
        return out

    async def run(self, inp: CodeAnalysisInput) -> CodeAnalysisOutput:
        inp = CodeAnalysisInput.model_validate(inp.model_dump())
        plog = self.log.bind(paper_id=inp.paper_id)
        warnings: list[str] = []

        try:
            tree = await github_tool.fetch_repo_tree(inp.repo_url)
        except Exception as exc:  # noqa: BLE001
            plog.error("repo_tree_failed", error=str(exc))
            tree = []

        py_files = [f for f in tree if f["path"].lower().endswith(".py")]
        if not py_files:
            plog.warning("no_python_code")
            fact = CodeFact(
                paper_id=inp.paper_id,
                repo_url=inp.repo_url,
                fact_id=str(uuid.uuid4()),
                fact_type="missing_eval",
                key="no_python_code",
                value=None,
                file_path="",
                evidence="No Python files found in repository.",
            )
            await queries.insert_code_fact(fact)
            await queries.update_paper_status(inp.paper_id, PaperStatus.CODE_ANALYZED)
            return CodeAnalysisOutput(
                paper_id=inp.paper_id,
                repo_url=inp.repo_url,
                facts_extracted=1,
                fact_ids=[fact.fact_id],
                files_analyzed=0,
                warnings=["no_python_code"],
            )

        fact_ids: list[str] = []
        files_analyzed = 0
        for f in tree[:_MAX_FILES]:
            try:
                content = await github_tool.fetch_file(f["download_url"])
            except Exception as exc:  # noqa: BLE001
                plog.warning("file_fetch_failed", path=f["path"], error=str(exc))
                continue
            if not content:
                continue
            files_analyzed += 1
            for chunk in self._chunks(content):
                try:
                    raw_facts = await self._analyze_file(
                        inp.repo_url, inp.paper_id, f["path"], chunk
                    )
                except Exception as exc:  # noqa: BLE001
                    plog.warning("file_analysis_failed", path=f["path"], error=str(exc))
                    warnings.append(f"file_failed:{f['path']}")
                    continue
                for rf in raw_facts:
                    fact = self._to_fact(inp.paper_id, inp.repo_url, f["path"], rf)
                    if fact is None:
                        continue
                    await queries.insert_code_fact(fact)
                    fact_ids.append(fact.fact_id)

        await queries.update_paper_status(inp.paper_id, PaperStatus.CODE_ANALYZED)
        return CodeAnalysisOutput(
            paper_id=inp.paper_id,
            repo_url=inp.repo_url,
            facts_extracted=len(fact_ids),
            fact_ids=fact_ids,
            files_analyzed=files_analyzed,
            warnings=warnings,
        )

    def _to_fact(
        self, paper_id: str, repo_url: str, default_path: str, rf: dict
    ) -> CodeFact | None:
        try:
            line_range = rf.get("line_range")
            if isinstance(line_range, (list, tuple)) and len(line_range) == 2:
                line_range = (int(line_range[0]), int(line_range[1]))
            else:
                line_range = None
            value = rf.get("value")
            return CodeFact(
                paper_id=paper_id,
                repo_url=repo_url,
                fact_id=str(uuid.uuid4()),
                fact_type=str(rf.get("fact_type", "hyperparameter")),
                key=str(rf["key"]),
                value=str(value) if value is not None else None,
                file_path=str(rf.get("file_path") or default_path),
                line_range=line_range,
                evidence=str(rf.get("evidence", "")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self.log.warning("fact_parse_failed", error=str(exc))
            return None

    async def _run_one(self, paper_id: str) -> None:
        paper = await queries.get_paper(paper_id)
        if paper is None:
            return
        plog = self.log.bind(paper_id=paper_id)
        if not paper.repo_url:
            plog.warning("no_repo_skip_code_analysis")
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)
            return
        inp = CodeAnalysisInput(paper_id=paper_id, repo_url=str(paper.repo_url))
        try:
            await self.run(inp)
        except Exception as exc:  # noqa: BLE001
            plog.error("code_analysis_failed", error=str(exc))
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)

    async def run_all(self, paper_ids: list[str]) -> None:
        await asyncio.gather(
            *(run_with_limit(self._run_one(pid)) for pid in paper_ids)
        )
