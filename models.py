# models.py

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, HttpUrl, field_validator


class PaperStatus(str, Enum):
    DISCOVERED = "discovered"
    REPO_RESOLVED = "repo_resolved"
    CLAIMS_EXTRACTED = "claims_extracted"
    CODE_ANALYZED = "code_analyzed"
    GAPS_ANALYZED = "gaps_analyzed"
    DONE = "done"
    FAILED = "failed"


class Paper(BaseModel):
    arxiv_id: str                          # e.g. "2106.09685"
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    pdf_url: HttpUrl
    arxiv_url: HttpUrl
    lora_variant_tag: str                  # e.g. "LoRA", "QLoRA", "AdaLoRA"
    status: PaperStatus = PaperStatus.DISCOVERED
    repo_url: Optional[HttpUrl] = None
    repo_confidence: Optional[float] = None  # 0.0–1.0
    citation_count: Optional[int] = None     # contradiction pivot weight


class BenchmarkClaim(BaseModel):
    paper_id: str                           # FK → Paper.arxiv_id
    claim_id: str                           # UUID
    metric: str                             # e.g. "BLEU", "accuracy", "perplexity"
    dataset: str                            # e.g. "GLUE/MNLI", "WinoGrande"
    model_base: str                         # e.g. "LLaMA-7B"
    reported_value: float
    unit: Optional[str] = None              # e.g. "%", "points"
    conditions: dict[str, str] = {}        # e.g. {"rank": "8", "lr": "3e-4"}
    is_conditional: bool = False
    claim_confidence: float = 1.0           # LLM self-assessed extraction confidence 0.0–1.0
    source_section: str                     # e.g. "Table 2", "Section 4.1"
    raw_text: str                           # verbatim sentence from paper


class CodeFact(BaseModel):
    paper_id: str
    repo_url: str
    fact_id: str                            # UUID
    fact_type: str                          # "hyperparameter", "dataset", "metric_logged", "missing_eval"
    key: str                                # e.g. "rank", "learning_rate"
    value: Optional[str] = None
    file_path: str
    line_range: Optional[tuple[int, int]] = None
    evidence: str                           # code snippet or comment


class ReproducibilityGap(BaseModel):
    gap_id: str                             # UUID
    paper_id: str
    claim_id: str                           # FK → BenchmarkClaim.claim_id
    fact_id: Optional[str] = None          # FK → CodeFact.fact_id
    gap_type: str                           # "missing_code", "value_mismatch", ...
    severity: str                           # "critical", "major", "minor"
    description: str
    paper_value: Optional[str] = None
    code_value: Optional[str] = None

    @field_validator("severity", mode="before")
    @classmethod
    def _guard_severity(cls, v: str) -> str:
        return v if v in {"critical", "major", "minor"} else "minor"


class Contradiction(BaseModel):
    contradiction_id: str                   # UUID
    paper_a_id: str
    paper_b_id: str
    claim_a_id: str
    claim_b_id: str
    contradiction_type: str                 # "direct_numeric", "conditional_flip", "dataset_scope"
    description: str
    severity: str                           # "high", "medium", "low"

    @field_validator("severity", mode="before")
    @classmethod
    def _guard_severity(cls, v: str) -> str:
        return v if v in {"high", "medium", "low"} else "low"


class AuditReport(BaseModel):
    generated_at: datetime
    papers_audited: int
    total_claims: int
    total_gaps: int
    total_contradictions: int
    critical_gaps: list[ReproducibilityGap]
    high_contradictions: list[Contradiction]
    summary_by_paper: list[dict]
    methodology_notes: str


class UserSessionOutput(BaseModel):
    research_question: str           # one-sentence distillation of user's goal
    variants_of_interest: list[str] | str   # specific variants or "all"
    benchmarks_of_interest: list[str] | str # specific benchmarks or "all"
    search_queries: list[str]        # 3-5 arXiv query strings
    raw_user_input: str              # original verbatim input (+ follow-up if any)
    ambiguous: bool = False          # True if WARNING "ambiguous_user_query" logged


# ── Agent I/O models ──────────────────────────────────────────────────────────

class PaperDiscoveryInput(BaseModel):
    query_terms: list[str]
    max_results_per_term: int = 50
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    variants_of_interest: list[str] | str = "all"


class PaperDiscoveryOutput(BaseModel):
    papers_found: int
    paper_ids: list[str]
    skipped: int


class RepoResolutionInput(BaseModel):
    paper_id: str
    title: str
    abstract: str
    pdf_text_first_2_pages: str


class RepoResolutionOutput(BaseModel):
    paper_id: str
    repo_url: Optional[str] = None
    confidence: float
    resolution_method: str


class ClaimExtractionInput(BaseModel):
    paper_id: str
    pdf_url: str
    full_text: str


class ClaimExtractionOutput(BaseModel):
    paper_id: str
    claims_extracted: int
    claim_ids: list[str]
    tables_parsed: int
    warnings: list[str] = []


class CodeAnalysisInput(BaseModel):
    paper_id: str
    repo_url: str


class CodeAnalysisOutput(BaseModel):
    paper_id: str
    repo_url: str
    facts_extracted: int
    fact_ids: list[str]
    files_analyzed: int
    warnings: list[str] = []


class GapAnalysisInput(BaseModel):
    paper_id: str
    claims: list[BenchmarkClaim]
    code_facts: list[CodeFact]


class GapAnalysisOutput(BaseModel):
    paper_id: str
    gaps_found: int
    gap_ids: list[str]
    severity_counts: dict[str, int]


class ContradictionMappingInput(BaseModel):
    all_claims: list[BenchmarkClaim]
    all_papers: list[Paper]


class ContradictionMappingOutput(BaseModel):
    contradictions_found: int
    contradiction_ids: list[str]
    papers_involved: list[str]


class ReportGenerationInput(BaseModel):
    output_dir: str
    include_raw_claims: bool = False
    severity_filter: Optional[str] = None


class ReportGenerationOutput(BaseModel):
    report_md_path: str
    report_html_path: str
    papers_in_report: int
    total_gaps: int
    total_contradictions: int
