import json
from typing import Literal

from pydantic import BaseModel

from core.llm_engine import LLMProviderError, ToolCall, ToolUsingLLMClient
from observability.tracing import ExecutionSummary, RunTrace
from tools.executor import ToolExecutor
from tools.registry import ToolRegistryError
from validation.schema import JewelryProduct, parse_and_validate

DEFAULT_MAX_AGENT_STEPS = 5
DEFAULT_MAX_TOOL_CALLS = 5


class AgentOrchestrationError(RuntimeError):
    """Raised when the bounded agent cannot safely produce a final response."""


class ToolExecutionSummary(BaseModel):
    tool_name: str
    result: dict


class SupportingSource(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    score: float


class AgentResult(BaseModel):
    proposal: JewelryProduct
    tools_used: list[str]
    supporting_information: list[ToolExecutionSummary]
    sources: list[SupportingSource]
    grounding_status: Literal["not_used", "grounded", "insufficient"]
    steps: int
    termination_reason: Literal["completed"]
    execution: ExecutionSummary


def _assistant_content(response_text: str | None, tool_calls: list[ToolCall]) -> list[dict]:
    content = []
    if response_text:
        content.append({"type": "text", "text": response_text})
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        }
        for call in tool_calls
    )
    return content


def _parse_final_product(text: str) -> JewelryProduct:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        return parse_and_validate(cleaned.strip())
    except ValueError as exc:
        raise AgentOrchestrationError("The model returned an invalid final proposal") from exc


def _grounding_from_summaries(
    summaries: list[ToolExecutionSummary],
) -> tuple[list[SupportingSource], Literal["not_used", "grounded", "insufficient"]]:
    retrieval_summaries = [
        summary for summary in summaries if summary.tool_name == "search_knowledge"
    ]
    if not retrieval_summaries:
        return [], "not_used"

    sources_by_chunk = {}
    for summary in retrieval_summaries:
        results = summary.result.get("results", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            try:
                source = SupportingSource.model_validate(result)
            except ValueError:
                continue
            sources_by_chunk[source.chunk_id] = source
    sources = list(sources_by_chunk.values())
    return sources, "grounded" if sources else "insufficient"


def run_agent(
    user_request: str,
    llm_client: ToolUsingLLMClient,
    tool_executor: ToolExecutor,
    run_id: str,
    max_steps: int = DEFAULT_MAX_AGENT_STEPS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    trace: RunTrace | None = None,
) -> AgentResult:
    if max_steps < 1 or max_tool_calls < 1:
        raise ValueError("Agent limits must be at least 1")

    messages: list[dict] = [{"role": "user", "content": user_request}]
    summaries: list[ToolExecutionSummary] = []
    active_trace = trace or RunTrace(run_id, "agent")

    for step in range(1, max_steps + 1):
        llm_started = active_trace.llm_call_started(step)
        try:
            response = llm_client.generate_with_tools(messages, tool_executor.definitions())
            active_trace.llm_call_completed(llm_started, step, response.usage)
        except LLMProviderError:
            active_trace.llm_call_failed(llm_started, step)
            active_trace.finish("provider_error", step, failed=True)
            raise

        if response.tool_calls:
            if len(summaries) + len(response.tool_calls) > max_tool_calls:
                active_trace.finish("maximum_tool_calls", step, failed=True)
                raise AgentOrchestrationError("The agent reached its tool-call limit")
            messages.append(
                {
                    "role": "assistant",
                    "content": _assistant_content(response.text, response.tool_calls),
                }
            )
            tool_results = []
            for call in response.tool_calls:
                tool_started = active_trace.tool_requested(call.name, step)
                try:
                    result = tool_executor.invoke(call.name, call.arguments)
                except ToolRegistryError as exc:
                    active_trace.tool_failed(call.name, step, tool_started)
                    active_trace.finish("tool_error", step, failed=True)
                    raise AgentOrchestrationError("A requested tool could not be executed") from exc
                active_trace.tool_completed(call.name, step, tool_started)
                if call.name == "search_knowledge":
                    active_trace.retrieval_completed(
                        step,
                        result.get("result_count", 0)
                        if isinstance(result, dict)
                        else 0,
                    )
                summaries.append(ToolExecutionSummary(tool_name=call.name, result=result))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.text:
            try:
                proposal = _parse_final_product(response.text)
            except AgentOrchestrationError:
                active_trace.validation_failed(step)
                active_trace.finish("invalid_final_response", step, failed=True)
                raise
            sources, grounding_status = _grounding_from_summaries(summaries)
            active_trace.finish("completed", step)
            return AgentResult(
                proposal=proposal,
                tools_used=[summary.tool_name for summary in summaries],
                supporting_information=summaries,
                sources=sources,
                grounding_status=grounding_status,
                steps=step,
                termination_reason="completed",
                execution=active_trace.summary(),
            )

        active_trace.finish("empty_response", step, failed=True)
        raise AgentOrchestrationError("The model returned neither a tool call nor a final response")

    active_trace.finish("maximum_steps", max_steps, failed=True)
    raise AgentOrchestrationError("The agent reached its maximum step limit")
