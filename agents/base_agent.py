# agents/base_agent.py

from __future__ import annotations
import asyncio
import functools
import logging
import random
import sys
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

from config import settings

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.pipeline_concurrency)
    return _semaphore


async def run_with_limit(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Await *coro* inside the shared pipeline semaphore.
    Cap is PIPELINE_CONCURRENCY from .env (default 3).
    """
    async with _get_semaphore():
        return await coro


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    jitter_max: float = 1.0,
    retriable: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[_F], _F]:
    """
    Async function decorator — retries up to *max_attempts* times on any
    exception in *retriable*.

    Back-off formula:  sleep = backoff_base ** attempt + uniform(0, jitter_max)
    """
    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_logger(fn.__module__)
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retriable as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = backoff_base ** attempt + random.uniform(0, jitter_max)
                        log.warning(
                            "retry",
                            fn=fn.__qualname__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay_s=round(delay, 2),
                            exc=str(exc),
                        )
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for *name*."""
    return structlog.get_logger(name)


# ── One-time structlog configuration ─────────────────────────────────────────

def _configure_structlog() -> None:
    level_str = getattr(settings, "log_level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if level == logging.DEBUG
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")


_configure_structlog()
