# agents/claim_extraction.py

from __future__ import annotations
import asyncio
import uuid

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger, run_with_limit
from agents.llm import call_llm_json, make_client
from config import settings
from db import queries
from models import (
    BenchmarkClaim,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    PaperStatus,
)
from tools import arxiv_tool

log = get_logger(__name__)

_SYSTEM = """You are a scientific claim extractor. Given the full text of an ML paper, identify
every quantitative benchmark claim — any statement that reports a numeric result on
a named dataset with a named metric. Include claims from tables, figures captions,
and prose.

For each claim extract:
- metric (e.g. "accuracy", "BLEU-4", "perplexity")
- dataset (e.g. "GLUE/SST-2", "MT-Bench")
- model_base (e.g. "LLaMA-7B", "RoBERTa-large")
- reported_value (numeric)
- unit (e.g. "%", "points", null)
- conditions: any hyperparameters or constraints stated alongside this claim (object of string->string)
- is_conditional: true if the claim only holds under specific conditions
- claim_confidence: your extraction confidence 0.0-1.0
- source_section: where in the paper this appears
- raw_text: the exact sentence or table cell

Return a JSON array of claim objects. Extract ALL claims, not just the best ones."""

_WINDOW_PAGES_CHARS = 30_000  # approx chars per sliding window


class ClaimExtractionAgent:
    """Agent 3 — extracts structured benchmark claims from a paper PDF."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def _extract_window(self, paper_id: str, text: str) -> list[dict]:
        user = f"Paper ID: {paper_id}\n\nFull paper text:\n{text}"
        data = await call_llm_json(self.client, _SYSTEM, user)
        if isinstance(data, dict):
            data = data.get("claims", [])
        return data if isinstance(data, list) else []

    @staticmethod
    def _windows(text: str) -> list[str]:
        if len(text) <= _WINDOW_PAGES_CHARS:
            return [text]
        overlap = 2_000
        out: list[str] = []
        start = 0
        while start < len(text):
            out.append(text[start : start + _WINDOW_PAGES_CHARS])
            start += _WINDOW_PAGES_CHARS - overlap
        return out

    async def run(self, inp: ClaimExtractionInput) -> ClaimExtractionOutput:
        inp = ClaimExtractionInput.model_validate(inp.model_dump())
        plog = self.log.bind(paper_id=inp.paper_id)
        warnings: list[str] = []

        raw_claims: list[dict] = []
        for window in self._windows(inp.full_text):
            try:
                raw_claims.extend(await self._extract_window(inp.paper_id, window))
            except Exception as exc:  # noqa: BLE001
                plog.warning("claim_window_failed", error=str(exc))
                warnings.append(f"window_failed:{exc}")

        # Dedup by (metric, dataset, model_base, reported_value)
        seen: set[tuple] = set()
        claim_ids: list[str] = []
        for rc in raw_claims:
            claim = self._to_claim(inp.paper_id, rc)
            if claim is None:
                continue
            if claim.claim_confidence < settings.claim_min_confidence:
                continue
            key = (claim.metric, claim.dataset, claim.model_base, claim.reported_value)
            if key in seen:
                plog.info("duplicate_claim_skipped", key=str(key))
                continue
            seen.add(key)
            await queries.insert_claim(claim)
            claim_ids.append(claim.claim_id)

        if not claim_ids:
            plog.warning("no_claims_found")
            await queries.update_paper_status(inp.paper_id, PaperStatus.FAILED)
        else:
            await queries.update_paper_status(inp.paper_id, PaperStatus.CLAIMS_EXTRACTED)

        return ClaimExtractionOutput(
            paper_id=inp.paper_id,
            claims_extracted=len(claim_ids),
            claim_ids=claim_ids,
            tables_parsed=0,
            warnings=warnings,
        )

    def _to_claim(self, paper_id: str, rc: dict) -> BenchmarkClaim | None:
        try:
            conditions = rc.get("conditions") or {}
            if isinstance(conditions, dict):
                conditions = {str(k): str(v) for k, v in conditions.items()}
            else:
                conditions = {}
            return BenchmarkClaim(
                paper_id=paper_id,
                claim_id=str(uuid.uuid4()),
                metric=str(rc["metric"]),
                dataset=str(rc["dataset"]),
                model_base=str(rc.get("model_base", "unknown")),
                reported_value=float(rc["reported_value"]),
                unit=rc.get("unit"),
                conditions=conditions,
                is_conditional=bool(rc.get("is_conditional", False)),
                claim_confidence=float(rc.get("claim_confidence", 1.0)),
                source_section=str(rc.get("source_section", "unknown")),
                raw_text=str(rc.get("raw_text", "")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self.log.warning("claim_parse_failed", error=str(exc))
            return None

    async def _run_one(self, paper_id: str) -> None:
        paper = await queries.get_paper(paper_id)
        if paper is None or paper.status == PaperStatus.FAILED:
            return
        plog = self.log.bind(paper_id=paper_id)
        try:
            full_text = await arxiv_tool.fetch_paper_text(str(paper.pdf_url), full=True)
        except Exception as exc:  # noqa: BLE001
            plog.error("pdf_fetch_failed", error=str(exc))
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)
            return
        if not full_text:
            plog.error("empty_pdf_text")
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)
            return
        inp = ClaimExtractionInput(
            paper_id=paper_id, pdf_url=str(paper.pdf_url), full_text=full_text
        )
        try:
            await self.run(inp)
        except Exception as exc:  # noqa: BLE001
            plog.error("claim_extraction_failed", error=str(exc))
            await queries.update_paper_status(paper_id, PaperStatus.FAILED)

    async def run_all(self, paper_ids: list[str]) -> None:
        await asyncio.gather(
            *(run_with_limit(self._run_one(pid)) for pid in paper_ids)
        )
