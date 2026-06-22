# mcp_server/server.py
#
# Server bootstrap + tool registration. The MCP server is an INTERFACE layer:
# every handler delegates to services.py (service-first). Core orchestration
# remains LangGraph.

from __future__ import annotations
from collections.abc import Awaitable, Callable

from agents.base_agent import get_logger
from mcp_server.tools import (
    claims as claims_tool,
    code as code_tool,
    contradictions as contradictions_tool,
    gaps as gaps_tool,
    papers as papers_tool,
    pipeline as pipeline_tool,
    report as report_tool,
)

log = get_logger(__name__)

# Canonical tool registry. Each handler signature: (request: dict, token: str|None) -> dict
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict]]] = {
    "start_pipeline_run": pipeline_tool.start_pipeline_run,
    "get_run_status": pipeline_tool.get_run_status,
    "list_papers": papers_tool.list_papers,
    "get_claims": claims_tool.get_claims,
    "get_code_facts": code_tool.get_code_facts,
    "get_gaps": gaps_tool.get_gaps,
    "get_contradictions": contradictions_tool.get_contradictions,
    "generate_report": report_tool.generate_report,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "start_pipeline_run": "Start a full LoRA audit pipeline run (mutating; requires auth).",
    "get_run_status": "Get status/progress of a pipeline run by run_id.",
    "list_papers": "List all discovered/audited papers.",
    "get_claims": "Get benchmark claims, optionally filtered by paper_id.",
    "get_code_facts": "Get extracted code facts for a paper.",
    "get_gaps": "Get reproducibility gaps, optionally filtered by paper_id.",
    "get_contradictions": "Get all cross-paper contradictions.",
    "generate_report": "Generate the audit report MD+HTML (mutating; requires auth).",
}


async def dispatch(tool_name: str, request: dict, token: str | None = None) -> dict:
    """Dispatch a tool call through the registry. Used by tests and the MCP runtime."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    return await TOOL_REGISTRY[tool_name](request, token)


def build_mcp_server():  # pragma: no cover - requires the `mcp` package + runtime
    """
    Construct an MCP Server exposing TOOL_REGISTRY. Imported lazily so the rest of
    the system does not depend on the `mcp` package being installed.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    import json

    server = Server("claimcheck")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=TOOL_DESCRIPTIONS.get(name, name),
                inputSchema={"type": "object"},
            )
            for name in TOOL_REGISTRY
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        token = arguments.pop("token", None) if isinstance(arguments, dict) else None
        result = await dispatch(name, arguments or {}, token)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def main() -> None:  # pragma: no cover
    import asyncio

    from mcp.server.stdio import stdio_server

    async def _run() -> None:
        server = build_mcp_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    log.info("mcp_server_starting")
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
