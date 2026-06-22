# mcp_server/auth.py
#
# Token auth + per-tool rate limiting for mutating MCP tools.

from __future__ import annotations
import time
from collections import defaultdict, deque

from config import settings


class AuthError(Exception):
    """Raised when a mutating tool is called without a valid token."""


class RateLimitError(Exception):
    """Raised when a tool exceeds its per-minute rate limit."""


# Tools that mutate state require a valid MCP_AUTH_TOKEN.
MUTATING_TOOLS = {"start_pipeline_run", "generate_report"}

_WINDOW_SECONDS = 60.0
_call_log: dict[str, deque[float]] = defaultdict(deque)


def validate_token(tool_name: str, token: str | None) -> None:
    """Reject mutating tools without a valid token. Read tools are unauthenticated."""
    if tool_name not in MUTATING_TOOLS:
        return
    if not token or token != settings.mcp_auth_token:
        raise AuthError(f"Invalid or missing auth token for tool '{tool_name}'")


def check_rate_limit(tool_name: str, now: float | None = None) -> None:
    """Enforce a sliding-window per-tool rate limit (MCP_RATE_LIMIT_PER_MIN)."""
    now = time.monotonic() if now is None else now
    window = _call_log[tool_name]
    while window and now - window[0] > _WINDOW_SECONDS:
        window.popleft()
    if len(window) >= settings.mcp_rate_limit_per_min:
        raise RateLimitError(f"Rate limit exceeded for tool '{tool_name}'")
    window.append(now)


def reset_rate_limits() -> None:
    """Test helper — clear all rate-limit windows."""
    _call_log.clear()
