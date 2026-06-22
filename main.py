# main.py

from __future__ import annotations
import asyncio
from typing import TypedDict

import typer
from langgraph.graph import END, StateGraph

from agents.base_agent import get_logger
from agents.claim_extraction import ClaimExtractionAgent
from agents.code_analysis import CodeAnalysisAgent
from agents.contradiction_mapping import ContradictionMappingAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.paper_discovery import PaperDiscoveryAgent
from agents.report_generation import ReportGenerationAgent
from agents.repo_resolution import RepoResolutionAgent
from config import settings
from db import init_db
from session.user_session import run_user_session

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


# ── Shared pipeline state ─────────────────────────────────────────────────────
class GraphState(TypedDict):
    query_terms: list[str]
    variants_of_interest: list[str] | str
    benchmarks_of_interest: list[str] | str
    research_question: str
    paper_ids: list[str]
    current_paper_id: str | None
    papers_processed: int
    errors: list[str]
    report_path: str | None


# ── Node functions (one per agent) ───────────────────────────────────────────
async def node_discover(state: GraphState) -> GraphState:
    agent = PaperDiscoveryAgent()
    result = await agent.run(
        state["query_terms"], variants_of_interest=state["variants_of_interest"]
    )
    log.info("discovery_complete", found=result.papers_found)
    return {**state, "paper_ids": result.paper_ids}


async def node_resolve_repos(state: GraphState) -> GraphState:
    agent = RepoResolutionAgent()
    await agent.run_all(state["paper_ids"])
    return state


async def node_extract_claims(state: GraphState) -> GraphState:
    agent = ClaimExtractionAgent()
    await agent.run_all(state["paper_ids"])
    return state


async def node_analyze_code(state: GraphState) -> GraphState:
    agent = CodeAnalysisAgent()
    await agent.run_all(state["paper_ids"])
    return state


async def node_gap_analysis(state: GraphState) -> GraphState:
    agent = GapAnalysisAgent()
    await agent.run_all(state["paper_ids"])
    return state


async def node_contradiction_mapping(state: GraphState) -> GraphState:
    agent = ContradictionMappingAgent()
    await agent.run()
    return state


async def node_report(state: GraphState) -> GraphState:
    agent = ReportGenerationAgent()
    result = await agent.run(output_dir=str(settings.report_output_dir))
    return {**state, "report_path": result.report_md_path}


# ── Conditional edges ────────────────────────────────────────────────────────
def has_papers(state: GraphState) -> str:
    return "resolve" if state["paper_ids"] else END


def has_claims(state: GraphState) -> str:
    return "gap" if state["paper_ids"] else END


# ── Graph assembly ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("discover", node_discover)
    g.add_node("resolve", node_resolve_repos)
    g.add_node("extract", node_extract_claims)
    g.add_node("analyze", node_analyze_code)
    g.add_node("gap", node_gap_analysis)
    g.add_node("contradictions", node_contradiction_mapping)
    g.add_node("report", node_report)

    g.set_entry_point("discover")
    g.add_conditional_edges("discover", has_papers, {"resolve": "resolve", END: END})
    g.add_edge("resolve", "extract")
    g.add_edge("extract", "analyze")
    g.add_conditional_edges("analyze", has_claims, {"gap": "gap", END: END})
    g.add_edge("gap", "contradictions")
    g.add_edge("contradictions", "report")
    g.add_edge("report", END)

    return g


# ── CLI entrypoints ──────────────────────────────────────────────────────────
@app.command()
def run(
    max_papers: int = typer.Option(100, "--max"),
    output_dir: str = typer.Option("reports/", "--output"),
    resume: bool = typer.Option(False, "--resume", help="Resume from last checkpoint"),
):
    """Run the full audit pipeline (interactive session starts automatically)."""
    asyncio.run(_run_async(max_papers, output_dir, resume))


async def _run_async(max_papers: int, output_dir: str, resume: bool) -> None:
    settings.pipeline_max_papers = max_papers
    settings.report_output_dir = output_dir  # type: ignore[assignment]
    await init_db.create_tables()
    session_output = await run_user_session()

    graph = build_graph()

    initial_state: GraphState = {
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

    thread_id = {"configurable": {"thread_id": "main"}}

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as cp:
            compiled = graph.compile(checkpointer=cp)
            if resume:
                await compiled.ainvoke(None, thread_id)
            else:
                await compiled.ainvoke(initial_state, thread_id)
    except Exception as exc:  # noqa: BLE001 — checkpointer unavailable → run without it
        log.warning("checkpointer_unavailable", error=str(exc))
        compiled = graph.compile()
        await compiled.ainvoke(initial_state)

    log.info("pipeline_complete")


@app.command()
def ui():
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    )


if __name__ == "__main__":
    app()
