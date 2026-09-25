import json

from fastapi.testclient import TestClient

from api.routes import app, get_agent_llm_client, get_mcp_tool_executor
from core.llm_engine import ToolCall, ToolLLMResponse
from mcp_integration.client import MCPClientAdapter
from mcp_integration.executor import MCPToolExecutor
from mcp_integration.server import mcp_server
from tools.registry import create_default_registry


class FakeToolLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate_with_tools(self, messages, tools):
        return next(self.responses)


def set_fake_agent_llm(fake):
    app.dependency_overrides[get_agent_llm_client] = lambda: fake


def teardown_function():
    app.dependency_overrides.clear()


def test_agent_api_success(valid_product_dict):
    set_fake_agent_llm(
        FakeToolLLM(
            [
                ToolLLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="estimate_price_range",
                            arguments={
                                "category": "earrings",
                                "materials": ["gold"],
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ToolLLMResponse(
                    text=json.dumps(valid_product_dict), stop_reason="end_turn"
                ),
            ]
        )
    )

    response = TestClient(app).post(
        "/agent/generate",
        json={"description": "Design gold earrings and estimate their price"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["proposal"] == valid_product_dict
    assert body["tools_used"] == ["estimate_price_range"]
    assert body["supporting_information"][0]["result"]["is_estimate"] is True
    assert body["termination_reason"] == "completed"
    assert body["run_id"]
    assert body["execution"]["run_id"] == body["run_id"]
    assert body["execution"]["llm_calls"] == 2
    assert body["execution"]["tool_calls"] == 1
    assert body["execution"]["duration_ms"] >= 0
    assert body["execution"]["token_usage"] is None


def test_agent_api_failure_is_sanitized():
    set_fake_agent_llm(
        FakeToolLLM(
            [
                ToolLLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="unregistered_secret_tool",
                            arguments={},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ]
        )
    )

    response = TestClient(app).post(
        "/agent/generate", json={"description": "Use an unavailable tool"}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "agent_failed",
            "message": "Unable to complete the bounded generation workflow",
        }
    }
    assert "unregistered_secret_tool" not in response.text


def test_agent_api_mcp_execution_mode(valid_product_dict):
    set_fake_agent_llm(
        FakeToolLLM(
            [
                ToolLLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="lookup_material",
                            arguments={"material": "gold"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ToolLLMResponse(
                    text=json.dumps(valid_product_dict), stop_reason="end_turn"
                ),
            ]
        )
    )
    app.dependency_overrides[get_mcp_tool_executor] = lambda: MCPToolExecutor(
        create_default_registry(), MCPClientAdapter(mcp_server)
    )

    response = TestClient(app).post(
        "/agent/generate",
        json={
            "description": "Research gold earrings",
            "tool_execution": "mcp",
        },
    )

    assert response.status_code == 200
    assert response.json()["tools_used"] == ["lookup_material"]
