import json
from typing import Any

import pytest
from mcp.server import MCPServer

from core.llm_engine import ToolCall, ToolLLMResponse
from mcp_integration.client import MCPClientAdapter, MCPClientError, MCPToolInfo
from mcp_integration.executor import MCPToolExecutor
from mcp_integration.server import mcp_server
from pipeline.agent_orchestrator import AgentOrchestrationError, run_agent
from tools.executor import LocalToolExecutor
from tools.registry import ToolRegistry, create_default_registry


class FakeToolLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate_with_tools(self, messages, tools):
        return next(self.responses)


def tool_response(name, arguments):
    return ToolLLMResponse(
        tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


def test_mcp_server_lists_only_expected_tools():
    tools = MCPClientAdapter(mcp_server).list_tools()

    assert {tool.name for tool in tools} == {
        "lookup_material",
        "search_catalog",
        "estimate_price_range",
        "search_knowledge",
    }


def test_lookup_material_through_mcp():
    result = MCPClientAdapter(mcp_server).call_tool(
        "lookup_material", {"material": "rose gold"}
    )

    assert result["found"] is True
    assert result["material"]["name"] == "gold"


def test_search_catalog_through_mcp():
    result = MCPClientAdapter(mcp_server).call_tool(
        "search_catalog", {"query": "silver ring", "limit": 2}
    )

    assert result["match_count"] >= 1
    assert result["products"][0]["id"] == "J-1002"


def test_estimate_price_range_through_mcp():
    result = MCPClientAdapter(mcp_server).call_tool(
        "estimate_price_range",
        {"category": "earrings", "materials": ["gold"]},
    )

    assert result["is_estimate"] is True
    assert result["currency"] == "USD"


def test_search_knowledge_through_mcp():
    result = MCPClientAdapter(mcp_server).call_tool(
        "search_knowledge",
        {"query": "care for pearl jewelry", "top_k": 2},
    )

    assert result["insufficient_context"] is False
    assert result["results"][0]["document_id"] == "DOC-CARE-001"
    assert result["results"][0]["chunk_id"]


def test_invalid_mcp_arguments_are_rejected():
    with pytest.raises(MCPClientError):
        MCPClientAdapter(mcp_server).call_tool(
            "estimate_price_range",
            {"category": "spaceship", "materials": []},
        )


def test_unknown_mcp_tool_is_rejected():
    with pytest.raises(MCPClientError):
        MCPClientAdapter(mcp_server).call_tool("run_shell", {})


def test_mcp_client_normalizes_discovery_and_result():
    client = MCPClientAdapter(mcp_server)

    tool = next(tool for tool in client.list_tools() if tool.name == "lookup_material")
    result = client.call_tool("lookup_material", {"material": "ceramic"})

    assert isinstance(tool, MCPToolInfo)
    assert tool.input_schema["type"] == "object"
    assert isinstance(result, dict)
    assert result["material"]["name"] == "ceramic"


def test_bounded_agent_uses_mcp_backed_tool(valid_product_dict):
    executor = MCPToolExecutor(create_default_registry(), MCPClientAdapter(mcp_server))
    llm = FakeToolLLM(
        [
            tool_response("lookup_material", {"material": "gold"}),
            ToolLLMResponse(text=json.dumps(valid_product_dict), stop_reason="end_turn"),
        ]
    )

    result = run_agent("Research gold earrings", llm, executor, "mcp-run")

    assert result.tools_used == ["lookup_material"]
    assert result.supporting_information[0].result["found"] is True


def test_mcp_tool_failure_is_propagated_safely():
    failing_server = MCPServer("failing-test-server")

    @failing_server.tool(name="lookup_material", structured_output=True)
    def failing_lookup(material: str) -> dict[str, Any]:
        raise RuntimeError("private MCP server failure")

    executor = MCPToolExecutor(
        create_default_registry(), MCPClientAdapter(failing_server)
    )
    llm = FakeToolLLM([tool_response("lookup_material", {"material": "gold"})])

    with pytest.raises(AgentOrchestrationError) as error:
        run_agent("Research gold", llm, executor, "failure-run")

    assert "private MCP server failure" not in str(error.value)


def test_agent_cannot_use_non_allowlisted_discovered_tool():
    class AdvertisingClient:
        called = False

        def list_tools(self):
            return [MCPToolInfo("dangerous_tool", "Not approved", {"type": "object"})]

        def call_tool(self, name, arguments):
            self.called = True
            return {"unsafe": True}

    client = AdvertisingClient()
    executor = MCPToolExecutor(create_default_registry(), client)
    llm = FakeToolLLM([tool_response("dangerous_tool", {})])

    with pytest.raises(AgentOrchestrationError):
        run_agent("Try another tool", llm, executor, "allowlist-run")

    assert client.called is False
    assert "dangerous_tool" not in {tool["name"] for tool in executor.definitions()}


def test_non_allowlisted_retrieval_cannot_bypass_mcp_security():
    restricted_registry = ToolRegistry([])
    executor = MCPToolExecutor(restricted_registry, MCPClientAdapter(mcp_server))
    llm = FakeToolLLM(
        [tool_response("search_knowledge", {"query": "ceramic care"})]
    )

    with pytest.raises(AgentOrchestrationError):
        run_agent("Search care guidance", llm, executor, "restricted-rag-run")

    assert executor.definitions() == []


def test_existing_local_tool_execution_still_works():
    executor = LocalToolExecutor(create_default_registry())

    result = executor.invoke("lookup_material", {"material": "silver"})

    assert result["found"] is True
    assert result["material"]["name"] == "sterling silver"
