import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from observability.logging import log_event

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """Raised when MCP discovery or invocation cannot be normalized safely."""


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


def default_stdio_server() -> StdioServerParameters:
    project_root = Path(__file__).resolve().parents[1]
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_integration.server"],
        cwd=project_root,
    )


class MCPClientAdapter:
    """Small synchronous application adapter over the SDK's async MCP client."""

    def __init__(self, server: Any | None = None):
        self._server = server or default_stdio_server()

    async def _list_tools(self) -> list[MCPToolInfo]:
        log_event(logger, logging.INFO, "mcp_connection_opened", operation="discovery")
        try:
            async with Client(self._server) as client:
                result = await client.list_tools()
                tools = [
                    MCPToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.input_schema,
                    )
                    for tool in result.tools
                ]
                log_event(
                    logger,
                    logging.INFO,
                    "mcp_discovery_completed",
                    tool_count=len(tools),
                )
                return tools
        except Exception as exc:
            log_event(logger, logging.WARNING, "mcp_discovery_failed")
            raise MCPClientError("MCP tool discovery failed") from exc
        finally:
            log_event(logger, logging.INFO, "mcp_connection_closed", operation="discovery")

    def list_tools(self) -> list[MCPToolInfo]:
        return asyncio.run(self._list_tools())

    async def _call_tool(self, name: str, arguments: dict) -> dict:
        log_event(
            logger,
            logging.INFO,
            "mcp_connection_opened",
            operation="tool_invocation",
            tool_name=name,
        )
        try:
            async with Client(self._server) as client:
                result = await client.call_tool(name, arguments)
                if result.is_error:
                    raise MCPClientError("MCP tool returned an error")
                if not isinstance(result.structured_content, dict):
                    raise MCPClientError("MCP tool returned invalid structured content")
                log_event(
                    logger,
                    logging.INFO,
                    "mcp_tool_completed",
                    tool_name=name,
                )
                return result.structured_content
        except MCPClientError:
            log_event(logger, logging.WARNING, "mcp_tool_failed", tool_name=name)
            raise
        except Exception as exc:
            log_event(logger, logging.WARNING, "mcp_tool_failed", tool_name=name)
            raise MCPClientError("MCP tool invocation failed") from exc
        finally:
            log_event(
                logger,
                logging.INFO,
                "mcp_connection_closed",
                operation="tool_invocation",
                tool_name=name,
            )

    def call_tool(self, name: str, arguments: dict) -> dict:
        return asyncio.run(self._call_tool(name, arguments))
