# mcp_server/tools/contradictions.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import ContradictionsRequest


async def get_contradictions(request: dict, token: str | None = None) -> dict:
    """Read tool — returns all cross-paper contradictions."""
    auth.check_rate_limit("get_contradictions")
    ContradictionsRequest.model_validate(request)

    async def _handler() -> dict:
        contras = await services.get_contradictions()
        return {"contradictions": contras, "count": len(contras)}

    return await run_manager.audited_call("get_contradictions", request, _handler)
