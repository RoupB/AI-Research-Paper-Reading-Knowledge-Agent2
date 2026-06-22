# mcp_server/tools/gaps.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import GapsRequest


async def get_gaps(request: dict, token: str | None = None) -> dict:
    """Read tool — returns reproducibility gaps, optionally filtered by paper_id."""
    auth.check_rate_limit("get_gaps")
    req = GapsRequest.model_validate(request)

    async def _handler() -> dict:
        gaps = await services.get_gaps(req.paper_id)
        return {"gaps": gaps, "count": len(gaps)}

    return await run_manager.audited_call("get_gaps", request, _handler)
