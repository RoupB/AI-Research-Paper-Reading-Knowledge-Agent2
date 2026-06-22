# services.py
#
# Service-first architecture: the canonical, reusable business operations.
# Both the LangGraph pipeline (CLI) and the MCP server call into this layer,
# so agent logic is never duplicated in interface handlers.

from __future__ import annotations
import uuid

from agents.base_agent import get_logger
from db import init_db, queries
from models import PaperStatus

log = get_logger(__name__)


async def run_pipeline(
    research_question: str,
    variants: list[str] | str = "all",
    max_papers: int = 30,
    run_id: str | None = None,
    client=None,
) -> dict:
    """
    Execute the full audit pipeline for a research question.
    Reuses the LangGraph graph as the canonical control flow.
    Returns a dict with run_id, report_path and final paper count.
    """
    from main import build_graph
    from session.user_session import run_user_session_from_text
    from config import settings

    run_id = run_id or str(uuid.uuid4())
    settings.pipeline_max_papers = max_papers
    await init_db.create_tables()
    await queries.insert_pipeline_run(run_id, research_question, variants, max_papers)
    await queries.update_pipeline_run(run_id, status="running", current_stage="discover")

    try:
        session_output = await run_user_session_from_text(
            research_question, variants=variants, client=client
        )
        graph = build_graph().compile()
        state = {
            "query_terms": session_output.search_queries,
            "variants_of_interest": session_output.variants_of_interest,
            "benchmarks_of_interest": session_output.benchmarks_of_interest,
            "research_question": session_output.research_question,
            "paper_ids": [],
            "current_paper_id": None,
            "papers_processed": 0,
            "errors": [],
            "report_path": None,
        }
        result = await graph.ainvoke(state)
        await queries.update_pipeline_run(
            run_id, status="done", current_stage="report", progress=1.0
        )
        return {
            "run_id": run_id,
            "report_path": result.get("report_path"),
            "paper_ids": result.get("paper_ids", []),
        }
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline_service_failed", run_id=run_id, error=str(exc))
        await queries.update_pipeline_run(run_id, status="error", error=str(exc))
        raise


async def get_run_status(run_id: str) -> dict | None:
    return await queries.get_pipeline_run(run_id)


async def list_papers() -> list[dict]:
    papers = await queries.get_all_papers()
    return [p.model_dump(mode="json") for p in papers]


async def get_claims(paper_id: str | None = None) -> list[dict]:
    claims = (
        await queries.get_claims_by_paper(paper_id)
        if paper_id
        else await queries.get_all_claims()
    )
    return [c.model_dump(mode="json") for c in claims]


async def get_code_facts(paper_id: str) -> list[dict]:
    facts = await queries.get_facts_by_paper(paper_id)
    return [f.model_dump(mode="json") for f in facts]


async def get_gaps(paper_id: str | None = None) -> list[dict]:
    gaps = (
        await queries.get_gaps_by_paper(paper_id)
        if paper_id
        else await queries.get_all_gaps()
    )
    return [g.model_dump(mode="json") for g in gaps]


async def get_contradictions() -> list[dict]:
    contras = await queries.get_all_contradictions()
    return [c.model_dump(mode="json") for c in contras]


async def get_stats() -> dict:
    return await queries.get_audit_stats()


async def generate_report(
    output_dir: str | None = None,
    include_raw_claims: bool = False,
    client=None,
) -> dict:
    from agents.report_generation import ReportGenerationAgent

    agent = ReportGenerationAgent(client=client)
    result = await agent.run(
        output_dir=output_dir, include_raw_claims=include_raw_claims
    )
    return result.model_dump()
