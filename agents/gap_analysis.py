# agents/gap_analysis.py

from __future__ import annotations
import asyncio
import json
import uuid

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger, run_with_limit
from agents.llm import call_llm_json, make_client
from db import queries
from models import (
    GapAnalysisInput,
    GapAnalysisOutput,
    PaperStatus,
    ReproducibilityGap,
)

log = get_logger(__name__)

_VALID_SEVERITY = {"critical", "major", "minor"}

_SYSTEM = """You are a reproducibility auditor. You will receive structured benchmark claims
from a research paper and structured facts extracted from its code repository.

Identify all reproducibility gaps — cases where:
- A claimed metric is not computed in the code (missing_code)
- A hyperparameter value in the paper differs from the code default (value_mismatch)
- The dataset used in evaluation differs (dataset_mismatch)
- A condition stated in the paper (e.g. "rank=8") is not set anywhere in the code (condition_undisclosed)
- A metric is claimed but no logging or saving of that metric exists (metric_not_implemented)

Rate severity:
- critical: the claim cannot be reproduced at all from the code
- major: significant effort needed to reproduce; key parameter missing or wrong
- minor: cosmetic or likely recoverable discrepancy

Return a JSON array of gap objects with fields:
gap_type, severity, description, paper_value, code_value, claim_id, fact_id (nullable)."""


class GapAnalysisAgent:
    """Agent 5 — reconciles claims against code facts into reproducibility gaps."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or make_client()
        self.log = log

    async def run(self, inp: GapAnalysisInput) -> GapAnalysisOutput:
        inp = GapAnalysisInput.model_validate(inp.model_dump())
        plog = self.log.bind(paper_id=inp.paper_id)
        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        gap_ids: list[str] = []

        # Empty claims or code facts → single critical missing_code gap
        if not inp.claims or not inp.code_facts:
            claim_id = inp.claims[0].claim_id if inp.claims else "none"
            gap = ReproducibilityGap(
                gap_id=str(uuid.uuid4()),
                paper_id=inp.paper_id,
                claim_id=claim_id,
                fact_id=None,
                gap_type="missing_code",
                severity="critical",
                description="No claims or code facts available to reconcile; "
                "the paper's results cannot be reproduced from released code.",
            )
            await queries.insert_gap(gap)
            severity_counts["critical"] += 1
            await queries.update_paper_status(inp.paper_id, PaperStatus.GAPS_ANALYZED)
            return GapAnalysisOutput(
                paper_id=inp.paper_id,
                gaps_found=1,
                gap_ids=[gap.gap_id],
                severity_counts=severity_counts,
            )

        valid_claim_ids = {c.claim_id for c in inp.claims}
        valid_fact_ids = {f.fact_id for f in inp.code_facts}

        claims_json = json.dumps([c.model_dump() for c in inp.claims], default=str)
        facts_json = json.dumps([f.model_dump() for f in inp.code_facts], default=str)
        user = (
            f"Paper ID: {inp.paper_id}\n\nCLAIMS:\n{claims_json}\n\n"
            f"CODE FACTS:\n{facts_json}\n\nIdentify all reproducibility gaps."
        )

        try:
            data = await call_llm_json(self.client, _SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            plog.error("gap_llm_failed", error=str(exc))
            await queries.update_paper_status(inp.paper_id, PaperStatus.GAPS_ANALYZED)
            return GapAnalysisOutput(
                paper_id=inp.paper_id, gaps_found=0, gap_ids=[], severity_counts=severity_counts
            )
        if isinstance(data, dict):
            data = data.get("gaps", [])

        for rg in data if isinstance(data, list) else []:
            claim_id = rg.get("claim_id")
            if claim_id not in valid_claim_ids:
                plog.warning("invalid_claim_id_discarded", claim_id=claim_id)
                claim_id = next(iter(valid_claim_ids))
            fact_id = rg.get("fact_id")
            if fact_id is not None and fact_id not in valid_fact_ids:
                fact_id = None
            severity = rg.get("severity", "minor")
            if severity not in _VALID_SEVERITY:
                plog.warning("invalid_severity_coerced", severity=severity)
                severity = "minor"
            try:
                gap = ReproducibilityGap(
                    gap_id=str(uuid.uuid4()),
                    paper_id=inp.paper_id,
                    claim_id=claim_id,
                    fact_id=fact_id,
                    gap_type=str(rg.get("gap_type", "missing_code")),
                    severity=severity,
                    description=str(rg.get("description", "")),
                    paper_value=_opt_str(rg.get("paper_value")),
                    code_value=_opt_str(rg.get("code_value")),
                )
            except Exception as exc:  # noqa: BLE001
                plog.warning("gap_parse_failed", error=str(exc))
                continue
            await queries.insert_gap(gap)
            gap_ids.append(gap.gap_id)
            severity_counts[gap.severity] += 1

        await queries.update_paper_status(inp.paper_id, PaperStatus.GAPS_ANALYZED)
        return GapAnalysisOutput(
            paper_id=inp.paper_id,
            gaps_found=len(gap_ids),
            gap_ids=gap_ids,
            severity_counts=severity_counts,
        )

    async def _run_one(self, paper_id: str) -> None:
        paper = await queries.get_paper(paper_id)
        if paper is None:
            return
        plog = self.log.bind(paper_id=paper_id)
        claims = await queries.get_claims_by_paper(paper_id)
        facts = await queries.get_facts_by_paper(paper_id)
        inp = GapAnalysisInput(paper_id=paper_id, claims=claims, code_facts=facts)
        try:
            await self.run(inp)
            if paper.status != PaperStatus.FAILED:
                await queries.update_paper_status(paper_id, PaperStatus.DONE)
        except Exception as exc:  # noqa: BLE001
            plog.error("gap_analysis_failed", error=str(exc))

    async def run_all(self, paper_ids: list[str]) -> None:
        await asyncio.gather(
            *(run_with_limit(self._run_one(pid)) for pid in paper_ids)
        )


def _opt_str(v) -> str | None:
    return None if v is None else str(v)
