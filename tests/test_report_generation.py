# tests/test_report_generation.py

from __future__ import annotations

from pathlib import Path

import pytest

from agents.report_generation import ReportGenerationAgent
from db import queries
from models import PaperStatus, ReproducibilityGap


async def test_report_generation_writes_files(
    test_db, mock_llm, sample_paper, sample_claim, tmp_path
):
    await queries.insert_paper(sample_paper)
    await queries.update_paper_status(sample_paper.arxiv_id, PaperStatus.DONE)
    await queries.insert_claim(sample_claim)
    await queries.insert_gap(
        ReproducibilityGap(
            gap_id="g1",
            paper_id=sample_paper.arxiv_id,
            claim_id=sample_claim.claim_id,
            gap_type="value_mismatch",
            severity="critical",
            description="rank mismatch",
        )
    )

    client = mock_llm("Executive summary text.\n\nKey Findings\n- finding 1")
    agent = ReportGenerationAgent(client=client)
    out_dir = tmp_path / "reports"
    out = await agent.run(output_dir=str(out_dir), include_raw_claims=True)

    md = Path(out.report_md_path)
    html = Path(out.report_html_path)
    assert md.exists() and md.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0
    assert out.papers_in_report == 1
    assert out.total_gaps == 1

    text = md.read_text(encoding="utf-8")
    assert "LoRA Variants Research Audit Report" in text
    assert sample_paper.arxiv_id in text


async def test_empty_db_exits(test_db, mock_llm, tmp_path):
    client = mock_llm("summary")
    agent = ReportGenerationAgent(client=client)
    with pytest.raises(SystemExit):
        await agent.run(output_dir=str(tmp_path / "reports"))
