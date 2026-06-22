# tests/test_user_session.py

from __future__ import annotations

import pytest

from session import user_session
from models import UserSessionOutput

_GOOD = {
    "research_question": "Which LoRA variants work best for instruction tuning?",
    "variants_of_interest": ["LoRA", "QLoRA"],
    "benchmarks_of_interest": "all",
    "search_queries": ["LoRA instruction tuning", "QLoRA LLaMA"],
    "clarifying_question": None,
}


async def test_from_text_basic(mock_llm):
    client = mock_llm(_GOOD)
    out = await user_session.run_user_session_from_text("LoRA stuff", client=client)
    assert isinstance(out, UserSessionOutput)
    assert out.research_question == _GOOD["research_question"]
    assert out.search_queries == _GOOD["search_queries"]
    assert out.ambiguous is False


async def test_from_text_variants_override(mock_llm):
    client = mock_llm(_GOOD)
    out = await user_session.run_user_session_from_text(
        "LoRA stuff", variants=["DoRA"], client=client
    )
    assert out.variants_of_interest == ["DoRA"]


async def test_from_text_empty_queries_fallback(mock_llm):
    resp = dict(_GOOD, search_queries=[])
    client = mock_llm(resp)
    out = await user_session.run_user_session_from_text("vague", client=client)
    assert out.search_queries == ["LoRA fine-tuning"]


async def test_malformed_json_retry_then_succeed(mock_llm):
    import json

    client = mock_llm(["not json at all", json.dumps(_GOOD)])
    out = await user_session.run_user_session_from_text("LoRA", client=client)
    assert out.research_question == _GOOD["research_question"]


async def test_malformed_json_twice_raises(mock_llm):
    client = mock_llm(["nope", "still nope"])
    with pytest.raises(RuntimeError, match="session_llm_failed"):
        await user_session.run_user_session_from_text("LoRA", client=client)
