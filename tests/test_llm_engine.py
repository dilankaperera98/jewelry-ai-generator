from types import SimpleNamespace

from core.config import Settings
from core.llm_engine import AnthropicLLMClient


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
