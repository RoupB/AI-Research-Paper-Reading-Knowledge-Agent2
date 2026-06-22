# mcp_server/tools/report.py

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import GenerateReportRequest, GenerateReportResponse


async def generate_report(request: dict, token: str | None = None) -> dict:
    """Mutating tool — requires auth. Generates the audit report (MD + HTML)."""
    auth.validate_token("generate_report", token)
    auth.check_rate_limit("generate_report")
    req = GenerateReportRequest.model_validate(request)

    async def _handler() -> dict:
        result = await services.generate_report(
            output_dir=req.output_dir,
            include_raw_claims=req.include_raw_claims,
        )
        return GenerateReportResponse(**result).model_dump()

    return await run_manager.audited_call(
        "generate_report", request, _handler, run_id=req.run_id
    )
