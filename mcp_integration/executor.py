from mcp_integration.client import MCPClientAdapter, MCPClientError
from tools.registry import ToolExecutionError, ToolRegistry


class MCPToolExecutor:
    """Execute the local allowlist through an MCP client/server boundary."""

    def __init__(self, registry: ToolRegistry, client: MCPClientAdapter):
        self._registry = registry
        self._client = client

    def definitions(self) -> list[dict]:
        # The model sees the application's allowlist, never arbitrary discovery results.
        return self._registry.definitions()

    def invoke(self, name: str, raw_arguments: dict) -> dict:
        validated = self._registry.validate_arguments(name, raw_arguments)
        try:
            discovered_names = {tool.name for tool in self._client.list_tools()}
            if name not in discovered_names:
                raise ToolExecutionError("Approved tool is unavailable from MCP server")
            return self._client.call_tool(name, validated.model_dump(mode="json"))
        except ToolExecutionError:
            raise
        except MCPClientError as exc:
            raise ToolExecutionError("MCP tool execution failed") from exc
