# mcp_server/schemas.py
#
# Strict request/response validation for MCP tools.

from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


# ── Pipeline ──────────────────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    research_question: str = Field(..., min_length=1)
    variants: list[str] | str = "all"
    max_papers: int = Field(30, ge=1, le=500)


class StartRunResponse(BaseModel):
    run_id: str
    status: str


class RunStatusRequest(BaseModel):
    run_id: str = Field(..., min_length=1)


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    current_stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None


# ── Read tools ────────────────────────────────────────────────────────────────

class PaperFilter(BaseModel):
    pass


class ClaimsRequest(BaseModel):
    paper_id: Optional[str] = None


class CodeFactsRequest(BaseModel):
    paper_id: str = Field(..., min_length=1)


class GapsRequest(BaseModel):
    paper_id: Optional[str] = None


class ContradictionsRequest(BaseModel):
    pass


# ── Report ────────────────────────────────────────────────────────────────────

class GenerateReportRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    include_raw_claims: bool = False
    output_dir: Optional[str] = None


class GenerateReportResponse(BaseModel):
    report_md_path: str
    report_html_path: str
    papers_in_report: int
    total_gaps: int
    total_contradictions: int
