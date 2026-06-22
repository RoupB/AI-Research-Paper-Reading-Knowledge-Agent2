# session/user_session.py
#
# Interactive research scoping. Runs before the agent pipeline.

from __future__ import annotations
import json
import sys

from anthropic import AsyncAnthropic

from agents.base_agent import get_logger
from agents.llm import _strip_fences, call_llm, make_client
from models import UserSessionOutput

log = get_logger(__name__)

_SYSTEM = """You are a research scoping assistant for a system that audits LoRA-variant
machine learning papers.

Given the user's research interest, extract:
1. The core research question in one sentence
2. The specific LoRA variants they care about (or "all" if not specified)
3. The benchmarks or tasks they care about (or "all" if not specified)
4. A list of 3-5 arXiv search query strings that will find the most relevant papers
5. A clarifying question to ask the user IF their input is too vague to generate
   good search queries (set to null if input is clear enough)

Return ONLY JSON:
{
  "research_question": str,
  "variants_of_interest": list[str] | "all",
  "benchmarks_of_interest": list[str] | "all",
  "search_queries": list[str],
  "clarifying_question": str | null
}"""


async def _call_llm_for_session(
    user_input: str, client: AsyncAnthropic | None = None
) -> dict:
    """One LLM call returning the scoping dict. Retries once on malformed JSON."""
    client = client or make_client()
    user = f"User's research interest: {user_input}"
    response = await call_llm(client, _SYSTEM, user)
    try:
        return json.loads(_strip_fences(response))
    except json.JSONDecodeError:
        response = await call_llm(
            client, _SYSTEM, "Respond only with JSON, no prose\n" + user
        )
        try:
            return json.loads(_strip_fences(response))
        except json.JSONDecodeError as exc:
            raise RuntimeError("session_llm_failed") from exc


def _print(msg: str) -> None:
    print(msg, flush=True)


async def run_user_session(client: AsyncAnthropic | None = None) -> UserSessionOutput:
    """Full interactive scoping session (stdin prompts + confirmation)."""
    client = client or make_client()

    # Non-TTY (piped) input → read one line, run non-interactively
    if not sys.stdin.isatty():
        line = sys.stdin.readline().strip()
        if not line:
            _print("No input provided.")
            raise SystemExit(1)
        return await run_user_session_from_text(line, client=client)

    # Step 1 — Welcome
    _print(
        "Welcome to ClaimCheck — LoRA Research Audit System\n"
        "What research area or question are you investigating?\n"
        "Example: I want to understand which LoRA variants work best for "
        "instruction tuning of LLaMA models"
    )
    # Step 2 — Collect input
    raw = input("> ").strip()
    if not raw:
        raw = input("Please describe your research interest: ").strip()
        if not raw:
            _print("No research interest provided. Exiting.")
            raise SystemExit(1)

    accumulated = raw
    ambiguous = False

    # Step 3 — Clarification agent
    data = await _call_llm_for_session(accumulated, client=client)

    # Step 4 — Ask clarifying question once
    if data.get("clarifying_question"):
        _print(data["clarifying_question"])
        follow_up = input("> ").strip()
        accumulated = f"Original: {raw}\nFollow-up: {follow_up}"
        data = await _call_llm_for_session(accumulated, client=client)
        if data.get("clarifying_question"):
            ambiguous = True
            log.warning("ambiguous_user_query")

    queries = data.get("search_queries") or []
    if not queries:
        log.warning("empty_search_queries")
        queries = ["LoRA fine-tuning"]

    # Step 5 — Confirm
    _print(
        f"I'll search for papers on: {data['research_question']}\n"
        "Search queries I'll use:"
    )
    for q in queries:
        _print(f"  - {q}")
    _print(f"Variants of interest: {data['variants_of_interest']}\n")
    _print("Press Enter to start or type a correction:")
    correction = input("> ").strip()
    if correction:
        accumulated = f"{accumulated}\nCorrection: {correction}"
        data = await _call_llm_for_session(accumulated, client=client)
        queries = data.get("search_queries") or ["LoRA fine-tuning"]

    return UserSessionOutput(
        research_question=data["research_question"],
        variants_of_interest=data["variants_of_interest"],
        benchmarks_of_interest=data["benchmarks_of_interest"],
        search_queries=queries,
        raw_user_input=accumulated,
        ambiguous=ambiguous,
    )


async def run_user_session_from_text(
    user_text: str,
    variants: list[str] | str = "all",
    client: AsyncAnthropic | None = None,
) -> UserSessionOutput:
    """
    Non-interactive version of run_user_session().
    Runs Steps 3 + 6 only (no stdin, no confirmation). Used by the Streamlit UI.
    """
    data = await _call_llm_for_session(user_text, client=client)
    if variants != "all" and variants != ["All"]:
        data["variants_of_interest"] = variants
    return UserSessionOutput(
        research_question=data["research_question"],
        variants_of_interest=data["variants_of_interest"],
        benchmarks_of_interest=data["benchmarks_of_interest"],
        search_queries=data.get("search_queries") or ["LoRA fine-tuning"],
        raw_user_input=user_text,
        ambiguous=False,
    )
