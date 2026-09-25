import json
import logging

from core.llm_engine import LLMTextResponse, ToolLLMResponse
from observability.logging import StructuredJsonFormatter
from observability.tracing import RunTrace, TokenUsage
from pipeline.agent_orchestrator import run_agent
from pipeline.generator_pipeline import generate_product_schema
from tools.executor import LocalToolExecutor
from tools.registry import create_default_registry


class FakeGenerationLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompt):
        return next(self.responses)


class FakeAgentLLM:
    def __init__(self, response):
        self.response = response

    def generate_with_tools(self, messages, tools):
        return self.response


def test_generation_trace_counts_validation_and_repair(valid_product_dict):
    trace = RunTrace("generation-trace", "generation")
    llm = FakeGenerationLLM(["invalid-json", json.dumps(valid_product_dict)])

    generate_product_schema("gold earrings", llm, trace)
    summary = trace.summary()

    assert summary.llm_calls == 2
    assert summary.validation_failures == 1
    assert summary.repair_attempts == 1
    assert summary.termination_reason == "completed"
    assert summary.duration_ms >= 0
    assert summary.token_usage is None


def test_generation_trace_captures_provider_usage(valid_product_dict):
    trace = RunTrace("generation-usage", "generation")
    llm = FakeGenerationLLM(
        [
            LLMTextResponse(
                text=json.dumps(valid_product_dict),
                usage=TokenUsage(input_tokens=10, output_tokens=20),
            )
        ]
    )

    generate_product_schema("gold earrings", llm, trace)

    assert trace.summary().token_usage.total_tokens == 30


def test_agent_trace_captures_available_token_usage(valid_product_dict):
    trace = RunTrace("agent-trace", "agent")
    llm = FakeAgentLLM(
        ToolLLMResponse(
            text=json.dumps(valid_product_dict),
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=25, output_tokens=15),
        )
    )

    result = run_agent(
        "Design earrings",
        llm,
        LocalToolExecutor(create_default_registry()),
        trace.run_id,
        trace=trace,
    )

    assert result.execution.llm_calls == 1
    assert result.execution.tool_calls == 0
    assert result.execution.token_usage.input_tokens == 25
    assert result.execution.token_usage.output_tokens == 15
    assert result.execution.token_usage.total_tokens == 40


def test_structured_formatter_emits_safe_correlatable_json():
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="run_started",
        args=(),
        exc_info=None,
    )
    record.event = "run_started"
    record.event_data = {"run_id": "safe-run", "workflow": "agent"}

    payload = json.loads(StructuredJsonFormatter().format(record))

    assert payload["event"] == "run_started"
    assert payload["run_id"] == "safe-run"
    assert "prompt" not in payload
    assert "api_key" not in payload
