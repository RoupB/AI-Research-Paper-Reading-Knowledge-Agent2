# mcp_server/tools/code.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import CodeFactsRequest


async def get_code_facts(request: dict, token: str | None = None) -> dict:
    """Read tool — returns code facts for a given paper."""
    auth.check_rate_limit("get_code_facts")
    req = CodeFactsRequest.model_validate(request)

    async def _handler() -> dict:
        facts = await services.get_code_facts(req.paper_id)
        return {"code_facts": facts, "count": len(facts)}

    return await run_manager.audited_call("get_code_facts", request, _handler)
