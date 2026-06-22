# agents/paper_discovery.py

from __future__ import annotations
from datetime import datetime

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger, run_with_limit
from agents.llm import call_llm_json, make_client
from config import settings
from db import queries
from models import (
    Paper,
    PaperDiscoveryInput,
    PaperDiscoveryOutput,
    PaperStatus,
)
from tools import arxiv_tool

log = get_logger(__name__)

# Controlled vocabulary of LoRA variant tags (v1 taxonomy).
KNOWN_VARIANTS = {
    "lora", "qlora", "adalora", "dora", "lora+", "loraplus", "vera", "dylora",
    "loftq", "lora-fa", "lorafa", "glora", "rslora", "molora", "flora",
}

_SYSTEM = """You are a research librarian specializing in parameter-efficient fine-tuning (PEFT)
methods. You assess whether an arXiv paper is genuinely about a LoRA variant —
meaning it proposes, benchmarks, or substantially extends Low-Rank Adaptation or
a named derivative (QLoRA, AdaLoRA, DoRA, LoRA+, DyLoRA, LoftQ, etc.).

Respond ONLY with valid JSON matching this schema:
{"relevant": bool, "lora_variant_tag": str | null, "reason": str}"""


class PaperDiscoveryAgent:
    """Agent 1 — discovers LoRA-variant papers via an agentic refinement loop."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def _assess(self, title: str, abstract: str) -> dict:
        user = (
            f"Title: {title}\nAbstract: {abstract}\n\n"
            "Is this paper about a LoRA variant? If yes, what is the primary variant name?"
        )
        return await call_llm_json(self.client, _SYSTEM, user)

    @staticmethod
    def _normalise_tag(tag: str | None) -> str:
        if not tag:
            return "OTHER_LORA"
        key = tag.strip().lower()
        return tag.strip() if key in KNOWN_VARIANTS else "OTHER_LORA"

    @staticmethod
    def _widen(query_terms: list[str], round_idx: int) -> list[str]:
        if round_idx == 1:
            return query_terms + [f"{q} survey benchmark" for q in query_terms[:2]]
        return ["parameter efficient fine-tuning LoRA"]

    async def run(
        self,
        query_terms: list[str],
        variants_of_interest: list[str] | str = "all",
        max_results_per_term: int | None = None,
    ) -> PaperDiscoveryOutput:
        inp = PaperDiscoveryInput.model_validate(
            {
                "query_terms": query_terms,
                "max_results_per_term": max_results_per_term
                or settings.arxiv_max_results_per_query,
                "variants_of_interest": variants_of_interest,
            }
        )

        accepted: dict[str, str] = {}  # arxiv_id -> variant tag
        seen: set[str] = set()
        skipped = 0
        terms = inp.query_terms

        for round_idx in range(settings.discovery_max_rounds):
            for term in terms:
                try:
                    results = await arxiv_tool.search_arxiv(
                        term, max_results=inp.max_results_per_term
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.error("arxiv_search_failed", term=term, error=str(exc))
                    continue

                for meta in results:
                    aid = meta["arxiv_id"]
                    if aid in seen:
                        skipped += 1
                        continue
                    seen.add(aid)
                    if not meta.get("abstract"):
                        self.log.warning("empty_abstract", arxiv_id=aid)
                        skipped += 1
                        continue
                    try:
                        verdict = await self._assess(meta["title"], meta["abstract"])
                    except Exception as exc:  # noqa: BLE001
                        self.log.warning("relevance_check_failed", arxiv_id=aid, error=str(exc))
                        skipped += 1
                        continue
                    if not verdict.get("relevant"):
                        skipped += 1
                        continue
                    tag = self._normalise_tag(verdict.get("lora_variant_tag"))
                    await self._persist(meta, tag)
                    accepted[aid] = tag
                    if len(accepted) >= settings.pipeline_max_papers:
                        break
                if len(accepted) >= settings.pipeline_max_papers:
                    break

            if len(accepted) >= settings.discovery_min_papers or len(
                accepted
            ) >= settings.pipeline_max_papers:
                break
            if round_idx < settings.discovery_max_rounds - 1:
                terms = self._widen(inp.query_terms, round_idx + 1)
                self.log.info("discovery_widen", round=round_idx + 1, found=len(accepted))

        if len(accepted) < settings.discovery_min_papers:
            self.log.warning("discovery_below_threshold", found=len(accepted))

        return PaperDiscoveryOutput(
            papers_found=len(accepted),
            paper_ids=list(accepted.keys()),
            skipped=skipped,
        )

    async def _persist(self, meta: dict, tag: str) -> None:
        published = meta.get("published")
        try:
            paper = Paper(
                arxiv_id=meta["arxiv_id"],
                title=meta["title"],
                authors=meta["authors"],
                abstract=meta["abstract"],
                published=datetime.fromisoformat(published)
                if published
                else datetime.utcnow(),
                pdf_url=meta["pdf_url"],
                arxiv_url=meta["arxiv_url"],
                lora_variant_tag=tag,
                status=PaperStatus.DISCOVERED,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("paper_validation_failed", arxiv_id=meta.get("arxiv_id"), error=str(exc))
            return
        await queries.insert_paper(paper)

    async def run_all(self, query_terms: list[str]) -> PaperDiscoveryOutput:
        return await run_with_limit(self.run(query_terms))
