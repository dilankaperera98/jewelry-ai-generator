from uuid import uuid4

from core.llm_engine import LLMClient, LLMTextResponse
from observability.tracing import RunTrace
from validation.schema import parse_and_validate


class GenerationExhaustedError(ValueError):
    """Raised when every allowed response fails schema validation."""


def build_repair_prompt(
    original_input: str, previous_output: str, validation_error: str
) -> str:
    """Build a correction request using the previous failed response."""
    return f"""Repair the previous JSON response for the original jewelry request.

Original request:
<original_request>{original_input}</original_request>

Previous invalid output:
<invalid_output>{previous_output}</invalid_output>

Validation error:
<validation_error>{validation_error}</validation_error>

Repair only what is necessary to satisfy the required schema. Return only the corrected raw JSON object, with no explanation or Markdown."""


def generate_product_schema(
    user_input: str,
    llm_client: LLMClient,
    trace: RunTrace | None = None,
) -> dict:
    """
    Full pipeline with retry loop:
    Input → LLM → Validate → Retry if needed → Output
    """
    prompt = user_input
    last_error: ValueError | None = None
    active_trace = trace or RunTrace(str(uuid4()), "generation")

    for attempt in range(3):
        step = attempt + 1
        llm_started = active_trace.llm_call_started(step)
        try:
            llm_response = llm_client.generate(prompt)
            if isinstance(llm_response, LLMTextResponse):
                raw_json = llm_response.text
                usage = llm_response.usage
            else:
                raw_json = llm_response
                usage = None
            active_trace.llm_call_completed(llm_started, step, usage)
        except Exception:
            active_trace.llm_call_failed(llm_started, step)
            active_trace.finish("provider_error", step, failed=True)
            raise
        try:
            product = parse_and_validate(raw_json)
            active_trace.finish("completed", step)
            return product.model_dump()
        except ValueError as e:
            last_error = e
            active_trace.validation_failed(step)
            if attempt < 2:
                active_trace.repair_attempted(step)
                prompt = build_repair_prompt(user_input, raw_json, str(e))

    active_trace.finish("validation_exhausted", 3, failed=True)
    raise GenerationExhaustedError(
        "The LLM did not produce a valid product schema after 3 attempts"
    ) from last_error
