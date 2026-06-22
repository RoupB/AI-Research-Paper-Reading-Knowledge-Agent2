# mcp_server/tools/claims.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import ClaimsRequest


async def get_claims(request: dict, token: str | None = None) -> dict:
    """Read tool — returns benchmark claims, optionally filtered by paper_id."""
    auth.check_rate_limit("get_claims")
    req = ClaimsRequest.model_validate(request)

    async def _handler() -> dict:
        claims = await services.get_claims(req.paper_id)
        return {"claims": claims, "count": len(claims)}

    return await run_manager.audited_call("get_claims", request, _handler)
