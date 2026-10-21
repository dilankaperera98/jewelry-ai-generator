from copy import deepcopy
from types import SimpleNamespace

from core.config import Settings
from core.llm_engine import AnthropicLLMClient, _anthropic_input_schema
from mcp_integration.client import MCPClientAdapter
from mcp_integration.executor import MCPToolExecutor
from mcp_integration.server import mcp_server
from tools.executor import LocalToolExecutor
from tools.registry import create_default_registry


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        return self.response


class FakeAnthropic:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_anthropic_input_schema_only_removes_nested_integer_bounds():
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
            },
            "name": {"type": "string", "minLength": 1},
            "ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    original = deepcopy(schema)

    converted = _anthropic_input_schema(schema)

    assert converted["properties"]["nested"]["properties"]["count"] == {
        "type": "integer"
    }
    assert converted["properties"]["name"]["minLength"] == 1
    assert converted["properties"]["ratio"]["minimum"] == 0.0
    assert converted["properties"]["ratio"]["maximum"] == 1.0
    assert schema == original


def test_anthropic_adapter_uses_native_tool_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="call-1",
                name="lookup_material",
                input={"material": "gold"},
            )
        ],
        stop_reason="tool_use",
    )
    sdk_client = FakeAnthropic(response)
    client = AnthropicLLMClient(Settings(anthropic_api_key="test-key"), sdk_client)

    result = client.generate_with_tools(
        [{"role": "user", "content": "Look up gold"}],
        [
            {
                "name": "lookup_material",
                "description": "Look up a material",
                "input_schema": {"type": "object"},
            }
        ],
    )

    assert result.tool_calls[0].name == "lookup_material"
    assert result.tool_calls[0].arguments == {"material": "gold"}
    assert sdk_client.messages.arguments["tools"][0]["strict"] is True
    assert sdk_client.messages.arguments["messages"][0]["role"] == "user"


def test_anthropic_outbound_schemas_remove_unsupported_integer_bounds():
    response = SimpleNamespace(content=[], stop_reason="end_turn")
    sdk_client = FakeAnthropic(response)
    client = AnthropicLLMClient(Settings(anthropic_api_key="test-key"), sdk_client)
    registry = create_default_registry()
    local_executor = LocalToolExecutor(registry)
    mcp_executor = MCPToolExecutor(registry, MCPClientAdapter(mcp_server))

    for executor in (local_executor, mcp_executor):
        definitions = executor.definitions()
        client.generate_with_tools(
            [{"role": "user", "content": "Research a design"}], definitions
        )

        outbound = {
            tool["name"]: tool["input_schema"]
            for tool in sdk_client.messages.arguments["tools"]
        }
        assert "minimum" not in outbound["search_catalog"]["properties"]["limit"]
        assert "maximum" not in outbound["search_catalog"]["properties"]["limit"]
        assert "minimum" not in outbound["search_knowledge"]["properties"]["top_k"]
        assert "maximum" not in outbound["search_knowledge"]["properties"]["top_k"]

        # Provider conversion must not weaken the canonical validation schema.
        catalog_limit = definitions[1]["input_schema"]["properties"]["limit"]
        assert catalog_limit["minimum"] == 1
        assert catalog_limit["maximum"] == 5
        knowledge_top_k = definitions[3]["input_schema"]["properties"]["top_k"]
        assert knowledge_top_k["minimum"] == 1
        assert knowledge_top_k["maximum"] == 5
