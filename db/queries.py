# db/queries.py

from __future__ import annotations
import json
from datetime import datetime, timezone

import aiosqlite

from config import settings
from models import (
    BenchmarkClaim,
    CodeFact,
    Contradiction,
    Paper,
    PaperStatus,
    ReproducibilityGap,
)


async def _connect() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(str(settings.db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Row → model reconstruction ────────────────────────────────────────────────

def _row_to_paper(row: aiosqlite.Row) -> Paper:
    d = dict(row)
    d["authors"] = json.loads(d["authors"]) if d.get("authors") else []
    if d.get("repo_url") in ("", None):
        d["repo_url"] = None
    return Paper.model_validate(
        {
            "arxiv_id": d["arxiv_id"],
            "title": d["title"],
            "authors": d["authors"],
            "abstract": d["abstract"],
            "published": d["published"],
            "pdf_url": d["pdf_url"],
            "arxiv_url": d["arxiv_url"],
            "lora_variant_tag": d["lora_variant_tag"],
            "status": d["status"],
            "repo_url": d["repo_url"],
            "repo_confidence": d["repo_confidence"],
        }
    )


def _row_to_claim(row: aiosqlite.Row) -> BenchmarkClaim:
    d = dict(row)
    return BenchmarkClaim(
        paper_id=d["paper_id"],
        claim_id=d["claim_id"],
        metric=d["metric"],
        dataset=d["dataset"],
        model_base=d["model_base"],
        reported_value=d["reported_value"],
        unit=d["unit"],
        conditions=json.loads(d["conditions"]) if d["conditions"] else {},
        is_conditional=bool(d["is_conditional"]),
        source_section=d["source_section"],
        raw_text=d["raw_text"],
    )


def _row_to_fact(row: aiosqlite.Row) -> CodeFact:
    d = dict(row)
    line_range = None
    if d.get("line_range"):
        try:
            start, end = d["line_range"].split(",")
            line_range = (int(start), int(end))
        except (ValueError, AttributeError):
            line_range = None
    return CodeFact(
        paper_id=d["paper_id"],
        repo_url=d["repo_url"],
        fact_id=d["fact_id"],
        fact_type=d["fact_type"],
        key=d["key"],
        value=d["value"],
        file_path=d["file_path"],
        line_range=line_range,
        evidence=d["evidence"],
    )


def _row_to_gap(row: aiosqlite.Row) -> ReproducibilityGap:
    d = dict(row)
    return ReproducibilityGap(
        gap_id=d["gap_id"],
        paper_id=d["paper_id"],
        claim_id=d["claim_id"],
        fact_id=d["fact_id"],
        gap_type=d["gap_type"],
        severity=d["severity"],
        description=d["description"],
        paper_value=d["paper_value"],
        code_value=d["code_value"],
    )


def _row_to_contradiction(row: aiosqlite.Row) -> Contradiction:
    d = dict(row)
    return Contradiction(
        contradiction_id=d["contradiction_id"],
        paper_a_id=d["paper_a_id"],
        paper_b_id=d["paper_b_id"],
        claim_a_id=d["claim_a_id"],
        claim_b_id=d["claim_b_id"],
        contradiction_type=d["contradiction_type"],
        description=d["description"],
        severity=d["severity"],
    )


# ── Papers ────────────────────────────────────────────────────────────────────

async def insert_paper(paper: Paper) -> None:
    """INSERT OR IGNORE into `papers`."""
    now = _now()
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO papers
                (arxiv_id, title, authors, abstract, published, pdf_url, arxiv_url,
                 lora_variant_tag, status, repo_url, repo_confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.arxiv_id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.published.isoformat(),
                str(paper.pdf_url),
                str(paper.arxiv_url),
                paper.lora_variant_tag,
                paper.status.value,
                str(paper.repo_url) if paper.repo_url else None,
                paper.repo_confidence,
                now,
                now,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def update_paper_status(arxiv_id: str, status: PaperStatus) -> None:
    """Advance the pipeline status of a paper and refresh updated_at."""
    conn = await _connect()
    try:
        await conn.execute(
            "UPDATE papers SET status = ?, updated_at = ? WHERE arxiv_id = ?",
            (status.value, _now(), arxiv_id),
        )
        await conn.commit()
    finally:
        await conn.close()


async def update_paper_repo(
    arxiv_id: str, repo_url: str | None, confidence: float | None
) -> None:
    """Persist a resolved repository URL + confidence for a paper."""
    conn = await _connect()
    try:
        await conn.execute(
            "UPDATE papers SET repo_url = ?, repo_confidence = ?, updated_at = ? WHERE arxiv_id = ?",
            (repo_url, confidence, _now(), arxiv_id),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_papers_by_status(status: PaperStatus) -> list[Paper]:
    """Fetch all papers in a given pipeline state."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM papers WHERE status = ?", (status.value,)
        )
        rows = await cursor.fetchall()
        return [_row_to_paper(r) for r in rows]
    finally:
        await conn.close()


async def get_paper(arxiv_id: str) -> Paper | None:
    """Fetch a single paper by arxiv_id."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        )
        row = await cursor.fetchone()
        return _row_to_paper(row) if row else None
    finally:
        await conn.close()


# ── Benchmark Claims ──────────────────────────────────────────────────────────

async def insert_claim(claim: BenchmarkClaim) -> None:
    """INSERT OR IGNORE into `benchmark_claims`."""
    now = _now()
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO benchmark_claims
                (claim_id, paper_id, metric, dataset, model_base, reported_value,
                 unit, conditions, is_conditional, source_section, raw_text,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id,
                claim.paper_id,
                claim.metric,
                claim.dataset,
                claim.model_base,
                claim.reported_value,
                claim.unit,
                json.dumps(claim.conditions),
                1 if claim.is_conditional else 0,
                claim.source_section,
                claim.raw_text,
                now,
                now,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_claims_by_paper(paper_id: str) -> list[BenchmarkClaim]:
    """Fetch all claims for one paper."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM benchmark_claims WHERE paper_id = ?", (paper_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_claim(r) for r in rows]
    finally:
        await conn.close()


async def get_all_claims() -> list[BenchmarkClaim]:
    """Full claim corpus ordered by (paper_id, claim_id)."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM benchmark_claims ORDER BY paper_id, claim_id"
        )
        rows = await cursor.fetchall()
        return [_row_to_claim(r) for r in rows]
    finally:
        await conn.close()


# ── Code Facts ────────────────────────────────────────────────────────────────

async def insert_code_fact(fact: CodeFact) -> None:
    """INSERT OR IGNORE into `code_facts`."""
    now = _now()
    line_range = (
        f"{fact.line_range[0]},{fact.line_range[1]}" if fact.line_range else None
    )
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO code_facts
                (fact_id, paper_id, repo_url, fact_type, key, value,
                 file_path, line_range, evidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.fact_id,
                fact.paper_id,
                fact.repo_url,
                fact.fact_type,
                fact.key,
                fact.value,
                fact.file_path,
                line_range,
                fact.evidence,
                now,
                now,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_facts_by_paper(paper_id: str) -> list[CodeFact]:
    """Fetch all code facts for one paper."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM code_facts WHERE paper_id = ?", (paper_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_fact(r) for r in rows]
    finally:
        await conn.close()


# ── Reproducibility Gaps ──────────────────────────────────────────────────────

async def insert_gap(gap: ReproducibilityGap) -> None:
    """INSERT OR IGNORE into `reproducibility_gaps`."""
    now = _now()
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO reproducibility_gaps
                (gap_id, paper_id, claim_id, fact_id, gap_type, severity,
                 description, paper_value, code_value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gap.gap_id,
                gap.paper_id,
                gap.claim_id,
                gap.fact_id,
                gap.gap_type,
                gap.severity,
                gap.description,
                gap.paper_value,
                gap.code_value,
                now,
                now,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_gaps_by_paper(paper_id: str) -> list[ReproducibilityGap]:
    """Fetch all reproducibility gaps for one paper."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            "SELECT * FROM reproducibility_gaps WHERE paper_id = ?", (paper_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_gap(r) for r in rows]
    finally:
        await conn.close()


async def get_all_gaps() -> list[ReproducibilityGap]:
    """Full gap corpus ordered by (severity, paper_id)."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            """
            SELECT * FROM reproducibility_gaps ORDER BY
                CASE severity WHEN 'critical' THEN 0 WHEN 'major' THEN 1 ELSE 2 END,
                paper_id
            """
        )
        rows = await cursor.fetchall()
        return [_row_to_gap(r) for r in rows]
    finally:
        await conn.close()


# ── Contradictions ────────────────────────────────────────────────────────────

async def insert_contradiction(c: Contradiction) -> None:
    """INSERT OR IGNORE into `contradictions`."""
    now = _now()
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO contradictions
                (contradiction_id, paper_a_id, paper_b_id, claim_a_id, claim_b_id,
                 contradiction_type, description, severity, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.contradiction_id,
                c.paper_a_id,
                c.paper_b_id,
                c.claim_a_id,
                c.claim_b_id,
                c.contradiction_type,
                c.description,
                c.severity,
                now,
                now,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_all_contradictions() -> list[Contradiction]:
    """Full contradiction corpus ordered by (severity, contradiction_id)."""
    conn = await _connect()
    try:
        cursor = await conn.execute(
            """
            SELECT * FROM contradictions ORDER BY
                CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                contradiction_id
            """
        )
        rows = await cursor.fetchall()
        return [_row_to_contradiction(r) for r in rows]
    finally:
        await conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────

async def get_all_papers() -> list[Paper]:
    """Fetch all papers regardless of status, newest first."""
    conn = await _connect()
    try:
        cursor = await conn.execute("SELECT * FROM papers ORDER BY published DESC")
        rows = await cursor.fetchall()
        return [_row_to_paper(r) for r in rows]
    finally:
        await conn.close()


# ── MCP run management ────────────────────────────────────────────────────────

async def insert_pipeline_run(
    run_id: str,
    research_question: str,
    variants: list[str] | str,
    max_papers: int,
) -> None:
    """Create a pipeline_runs row in 'queued' state."""
    now = _now()
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO pipeline_runs
                (run_id, research_question, variants, max_papers, status,
                 current_stage, progress, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'queued', NULL, 0.0, NULL, ?, ?)
            """,
            (run_id, research_question, json.dumps(variants), max_papers, now, now),
        )
        await conn.commit()
    finally:
        await conn.close()


async def update_pipeline_run(
    run_id: str,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    """Patch mutable fields on a pipeline_runs row."""
    sets, params = [], []
    if status is not None:
        sets.append("status = ?"); params.append(status)
    if current_stage is not None:
        sets.append("current_stage = ?"); params.append(current_stage)
    if progress is not None:
        sets.append("progress = ?"); params.append(progress)
    if error is not None:
        sets.append("error = ?"); params.append(error)
    sets.append("updated_at = ?"); params.append(_now())
    params.append(run_id)
    conn = await _connect()
    try:
        await conn.execute(
            f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE run_id = ?", params
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_pipeline_run(run_id: str) -> dict | None:
    """Fetch a pipeline_runs row as a dict."""
    conn = await _connect()
    try:
        cur = await conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None
    finally:
        await conn.close()


async def insert_mcp_tool_call(
    call_id: str,
    run_id: str | None,
    tool_name: str,
    request: str,
    response: str,
    status: str,
    latency_ms: float,
) -> None:
    """Persist a redacted MCP tool-call audit record."""
    conn = await _connect()
    try:
        await conn.execute(
            """
            INSERT OR IGNORE INTO mcp_tool_calls
                (call_id, run_id, tool_name, request, response, status, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (call_id, run_id, tool_name, request, response, status, latency_ms, _now()),
        )
        await conn.commit()
    finally:
        await conn.close()


# ── Aggregates ────────────────────────────────────────────────────────────────

async def get_audit_stats() -> dict:
    """Return a single dict of aggregated counts for the report generator."""
    conn = await _connect()
    try:
        async def _scalar(sql: str, params: tuple = ()) -> int:
            cur = await conn.execute(sql, params)
            row = await cur.fetchone()
            return row[0] if row else 0

        papers_total = await _scalar("SELECT COUNT(*) FROM papers")

        papers_by_status: dict[str, int] = {s.value: 0 for s in PaperStatus}
        cur = await conn.execute(
            "SELECT status, COUNT(*) FROM papers GROUP BY status"
        )
        for row in await cur.fetchall():
            papers_by_status[row[0]] = row[1]

        claims_total = await _scalar("SELECT COUNT(*) FROM benchmark_claims")
        claims_conditional = await _scalar(
            "SELECT COUNT(*) FROM benchmark_claims WHERE is_conditional = 1"
        )
        code_facts_total = await _scalar("SELECT COUNT(*) FROM code_facts")
        gaps_total = await _scalar("SELECT COUNT(*) FROM reproducibility_gaps")

        gaps_by_severity = {"critical": 0, "major": 0, "minor": 0}
        cur = await conn.execute(
            "SELECT severity, COUNT(*) FROM reproducibility_gaps GROUP BY severity"
        )
        for row in await cur.fetchall():
            gaps_by_severity[row[0]] = row[1]

        contradictions_total = await _scalar("SELECT COUNT(*) FROM contradictions")
        contradictions_by_severity = {"high": 0, "medium": 0, "low": 0}
        cur = await conn.execute(
            "SELECT severity, COUNT(*) FROM contradictions GROUP BY severity"
        )
        for row in await cur.fetchall():
            contradictions_by_severity[row[0]] = row[1]

        return {
            "papers_total": papers_total,
            "papers_by_status": papers_by_status,
            "claims_total": claims_total,
            "claims_conditional": claims_conditional,
            "code_facts_total": code_facts_total,
            "gaps_total": gaps_total,
            "gaps_by_severity": gaps_by_severity,
            "contradictions_total": contradictions_total,
            "contradictions_by_severity": contradictions_by_severity,
        }
    finally:
        await conn.close()
