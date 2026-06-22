# mcp_server/run_manager.py
#
# Run lifecycle + tool-call audit. Wraps every tool handler so latency, status,
# and redacted payloads are persisted to the DB.

from __future__ import annotations
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agents.base_agent import get_logger
from db import queries

log = get_logger(__name__)

_REDACT_KEYS = {"token", "auth_token", "api_key", "anthropic_api_key", "github_token"}


def _redact(payload: Any) -> str:
    """Serialise a payload to JSON with sensitive keys masked."""
    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: ("***" if k.lower() in _REDACT_KEYS else scrub(v))
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    try:
        return json.dumps(scrub(payload), default=str)[:8000]
    except (TypeError, ValueError):
        return str(payload)[:8000]


async def audited_call(
    tool_name: str,
    request: dict,
    handler: Callable[[], Awaitable[Any]],
    run_id: str | None = None,
) -> Any:
    """
    Execute *handler*, persisting an mcp_tool_calls audit record with latency and
    status. Re-raises any handler exception after recording it.
    """
    call_id = str(uuid.uuid4())
    start = time.monotonic()
    status = "success"
    response: Any = None
    try:
        response = await handler()
        return response
    except Exception as exc:  # noqa: BLE001
        status = "error"
        response = {"error": str(exc)}
        raise
    finally:
        latency_ms = (time.monotonic() - start) * 1000.0
        try:
            await queries.insert_mcp_tool_call(
                call_id=call_id,
                run_id=run_id,
                tool_name=tool_name,
                request=_redact(request),
                response=_redact(response),
                status=status,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("audit_persist_failed", tool=tool_name, error=str(exc))


def new_run_id() -> str:
    return str(uuid.uuid4())
