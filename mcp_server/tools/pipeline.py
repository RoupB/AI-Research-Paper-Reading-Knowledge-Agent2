# mcp_server/tools/pipeline.py
#
# Thin handlers mapping pipeline tools to service-layer calls.

from __future__ import annotations

import services
from mcp_server import auth, run_manager
from mcp_server.schemas import (
    RunStatusRequest,
    RunStatusResponse,
    StartRunRequest,
    StartRunResponse,
)


async def start_pipeline_run(request: dict, token: str | None = None) -> dict:
    """Mutating tool — requires auth. Starts a full audit pipeline run."""
    auth.validate_token("start_pipeline_run", token)
    auth.check_rate_limit("start_pipeline_run")
    req = StartRunRequest.model_validate(request)
    run_id = run_manager.new_run_id()

    async def _handler() -> dict:
        result = await services.run_pipeline(
            research_question=req.research_question,
            variants=req.variants,
            max_papers=req.max_papers,
            run_id=run_id,
        )
        return result

    await run_manager.audited_call(
        "start_pipeline_run", request, _handler, run_id=run_id
    )
    return StartRunResponse(run_id=run_id, status="done").model_dump()


async def get_run_status(request: dict, token: str | None = None) -> dict:
    """Read tool — returns the current status/progress of a run."""
    auth.check_rate_limit("get_run_status")
    req = RunStatusRequest.model_validate(request)

    async def _handler() -> dict:
        row = await services.get_run_status(req.run_id)
        if row is None:
            return RunStatusResponse(
                run_id=req.run_id, status="unknown"
            ).model_dump()
        return RunStatusResponse(
            run_id=req.run_id,
            status=row["status"],
            current_stage=row.get("current_stage"),
            progress=row.get("progress") or 0.0,
            error=row.get("error"),
        ).model_dump()

    return await run_manager.audited_call(
        "get_run_status", request, _handler, run_id=req.run_id
    )
