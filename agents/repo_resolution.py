# agents/repo_resolution.py

from __future__ import annotations
import asyncio

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger, run_with_limit
from agents.llm import call_llm_json, make_client
from config import settings
from db import queries
from models import (
    PaperStatus,
    RepoResolutionInput,
    RepoResolutionOutput,
)
from tools import arxiv_tool, github_tool

log = get_logger(__name__)

_SYSTEM = """You are an expert at locating the official code repository for ML research papers.
Given paper metadata and the first two pages of the PDF, extract any GitHub URLs
mentioned. If none are present, reason about likely repository names based on
paper title, author names, and institution.

Return ONLY JSON:
{
  "github_urls_found": [str],
  "most_likely_url": str | null,
  "confidence": float,
  "reasoning": str
}"""


class RepoResolutionAgent:
    """Agent 2 — resolves each paper's GitHub repository."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def run(self, inp: RepoResolutionInput) -> RepoResolutionOutput:
        inp = RepoResolutionInput.model_validate(inp.model_dump())
        plog = self.log.bind(paper_id=inp.paper_id)

        user = (
            f"Title: {inp.title}\nAuthors: \nAbstract: {inp.abstract}\n\n"
            f"PDF first pages:\n{inp.pdf_text_first_2_pages}\n\n"
            "Find the GitHub repository for this paper."
        )
        method = "pdf_link"
        try:
            data = await call_llm_json(self.client, _SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            plog.warning("repo_llm_failed", error=str(exc))
            data = {"most_likely_url": None, "confidence": 0.0}

        candidate = data.get("most_likely_url")
        if not candidate and data.get("github_urls_found"):
            candidate = data["github_urls_found"][0]
        if not candidate:
            method = "github_search"

        try:
            verified_url, confidence = await github_tool.resolve_repo(
                candidate, inp.title, []
            )
        except github_tool.ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            plog.warning("repo_resolution_error", error=str(exc))
            verified_url, confidence = None, 0.0

        if verified_url is None:
            method = "github_search"
        elif candidate and github_tool.normalize_repo_url(candidate) == verified_url:
            method = "pdf_link"
        else:
            method = "llm_inference"

        if confidence < 0.5:
            plog.warning("low_repo_confidence", confidence=confidence)
            verified_url = None

        await queries.update_paper_repo(inp.paper_id, verified_url, confidence)
        await queries.update_paper_status(inp.paper_id, PaperStatus.REPO_RESOLVED)

        return RepoResolutionOutput(
            paper_id=inp.paper_id,
            repo_url=verified_url,
            confidence=confidence,
            resolution_method=method,
        )

    async def _run_one(self, paper_id: str) -> None:
        paper = await queries.get_paper(paper_id)
        if paper is None:
            return
        plog = self.log.bind(paper_id=paper_id)
        try:
            first_pages = await arxiv_tool.fetch_paper_text(str(paper.pdf_url), pages=2)
        except Exception as exc:  # noqa: BLE001
            plog.warning("pdf_fetch_failed", error=str(exc))
            first_pages = ""

        inp = RepoResolutionInput(
            paper_id=paper_id,
            title=paper.title,
            abstract=paper.abstract,
            pdf_text_first_2_pages=first_pages,
        )
        try:
            await self.run(inp)
        except github_tool.ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            plog.error("repo_resolution_failed", error=str(exc))
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)

    async def run_all(self, paper_ids: list[str]) -> None:
        await asyncio.gather(
            *(run_with_limit(self._run_one(pid)) for pid in paper_ids)
        )
