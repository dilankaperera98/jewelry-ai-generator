import json

import pytest

from core.llm_engine import ToolCall, ToolLLMResponse
from pipeline.agent_orchestrator import AgentOrchestrationError, run_agent
from tools.domain_tools import MaterialLookupInput
from tools.registry import ToolDefinition, ToolRegistry, create_default_registry


class FakeToolLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_with_tools(self, messages, tools):
        self.calls.append({"messages": messages.copy(), "tools": tools})
        return next(self.responses)


def final_response(product):
    return ToolLLMResponse(text=json.dumps(product), stop_reason="end_turn")


def tool_response(call_id, name, arguments):
    return ToolLLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


def test_no_tool_path(valid_product_dict):
    llm = FakeToolLLM([final_response(valid_product_dict)])

    result = run_agent(
        "Design minimal gold earrings", llm, create_default_registry(), "run-1"
    )

    assert result.proposal.model_dump() == valid_product_dict
    assert result.tools_used == []
    assert result.steps == 1


def test_correct_tool_selection_flow(valid_product_dict):
    llm = FakeToolLLM(
        [
            tool_response("call-1", "lookup_material", {"material": "gold"}),
            final_response(valid_product_dict),
        ]
    )

    result = run_agent(
        "Use a supported durable material", llm, create_default_registry(), "run-2"
    )

    assert result.tools_used == ["lookup_material"]
    assert result.supporting_information[0].result["found"] is True
    assert {tool["name"] for tool in llm.calls[0]["tools"]} == {
        "lookup_material",
        "search_catalog",
        "estimate_price_range",
        "search_knowledge",
    }
    tool_result_message = llm.calls[1]["messages"][-1]
    assert tool_result_message["content"][0]["tool_use_id"] == "call-1"


def test_multiple_sequential_tool_calls(valid_product_dict):
    llm = FakeToolLLM(
        [
            tool_response("call-1", "lookup_material", {"material": "silver"}),
            tool_response("call-2", "search_catalog", {"query": "silver ring"}),
            final_response(valid_product_dict),
        ]
    )

    result = run_agent("Research a silver ring", llm, create_default_registry(), "run-3")

    assert result.tools_used == ["lookup_material", "search_catalog"]
    assert result.steps == 3


def test_agent_retrieval_produces_grounded_source_references(valid_product_dict):
    llm = FakeToolLLM(
        [
            tool_response(
                "call-1",
                "search_knowledge",
                {"query": "protect fragile ceramic jewelry during shipping", "top_k": 2},
            ),
            final_response(valid_product_dict),
        ]
    )

    result = run_agent(
        "Use packaging guidance for a ceramic design",
        llm,
        create_default_registry(),
        "grounded-run",
    )

    assert result.tools_used == ["search_knowledge"]
    assert result.grounding_status == "grounded"
    assert result.sources
    assert result.sources[0].document_id == "DOC-PACK-001"
    assert result.sources[0].chunk_id
    assert result.sources[0].score > 0


def test_agent_marks_insufficient_retrieval_context(valid_product_dict):
    llm = FakeToolLLM(
        [
            tool_response(
                "call-1",
                "search_knowledge",
                {"query": "quantum rocket propulsion telemetry", "top_k": 2},
            ),
            final_response(valid_product_dict),
        ]
    )

    result = run_agent(
        "Use unavailable knowledge",
        llm,
        create_default_registry(),
        "insufficient-run",
    )

    assert result.grounding_status == "insufficient"
    assert result.sources == []


def test_unknown_tool_terminates_agent():
    llm = FakeToolLLM([tool_response("call-1", "run_shell", {})])

    with pytest.raises(AgentOrchestrationError):
        run_agent("Unsafe request", llm, create_default_registry(), "run-4")


def test_invalid_tool_arguments_terminate_agent():
    llm = FakeToolLLM(
        [tool_response("call-1", "lookup_material", {"material": "   "})]
    )

    with pytest.raises(AgentOrchestrationError):
        run_agent("Look up nothing", llm, create_default_registry(), "run-5")


def test_tool_execution_failure_terminates_agent():
    def failing_tool(arguments):
        raise RuntimeError("do not leak")

    registry = ToolRegistry(
        [
            ToolDefinition(
                name="lookup_material",
                description="Test tool",
                input_model=MaterialLookupInput,
                implementation=failing_tool,
            )
        ]
    )
    llm = FakeToolLLM(
        [tool_response("call-1", "lookup_material", {"material": "gold"})]
    )

    with pytest.raises(AgentOrchestrationError):
        run_agent("Look up gold", llm, registry, "run-6")


def test_maximum_step_termination():
    llm = FakeToolLLM(
        [
            tool_response("call-1", "lookup_material", {"material": "gold"}),
            tool_response("call-2", "lookup_material", {"material": "gold"}),
        ]
    )

    with pytest.raises(AgentOrchestrationError, match="maximum step"):
        run_agent(
            "Keep looking", llm, create_default_registry(), "run-7", max_steps=2
        )
