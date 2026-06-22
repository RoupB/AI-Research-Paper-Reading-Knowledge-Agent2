# agents/llm.py
#
# Shared Anthropic client helpers. Agents use the `anthropic` SDK directly
# (not LangChain's wrapper) and every LLM call expects JSON-only output.

from __future__ import annotations
import json
import re
from typing import Any

from anthropic import AsyncAnthropic

from config import settings


def make_client() -> AsyncAnthropic:
    """Construct an AsyncAnthropic client from settings."""
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def call_llm(
    client: AsyncAnthropic,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Single Anthropic message call returning the raw text of the first block."""
    resp = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens or settings.anthropic_max_tokens,
        temperature=(
            temperature if temperature is not None else settings.anthropic_temperature
        ),
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


async def call_llm_json(
    client: AsyncAnthropic,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
) -> Any:
    """
    Call the LLM and parse JSON output, applying the mandatory retry pattern:
    on the first JSONDecodeError, retry once with a 'JSON only' instruction.
    A second failure raises JSONDecodeError for the caller to handle.
    """
    response = await call_llm(client, system, user, max_tokens=max_tokens)
    try:
        return json.loads(_strip_fences(response))
    except json.JSONDecodeError:
        response = await call_llm(
            client,
            system,
            "Respond only with JSON, no prose.\n" + user,
            max_tokens=max_tokens,
        )
        return json.loads(_strip_fences(response))
