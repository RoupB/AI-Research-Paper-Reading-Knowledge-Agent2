# mcp_server/tools/papers.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager


async def list_papers(request: dict, token: str | None = None) -> dict:
    """Read tool — returns all papers."""
    auth.check_rate_limit("list_papers")

    async def _handler() -> dict:
        papers = await services.list_papers()
        return {"papers": papers, "count": len(papers)}

    return await run_manager.audited_call("list_papers", request, _handler)
