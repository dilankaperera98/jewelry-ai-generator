import json

import pytest

from core.llm_engine import LLMProviderError
from pipeline.generator_pipeline import (
    GenerationExhaustedError,
    generate_product_schema,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_successful_generation_on_first_attempt(valid_product_dict):
    llm = FakeLLM([json.dumps(valid_product_dict)])

    result = generate_product_schema("gold earrings", llm)

    assert result == valid_product_dict
    assert llm.prompts == ["gold earrings"]


def test_invalid_first_response_followed_by_successful_repair(valid_product_dict):
    invalid_output = "{not-json}"
    llm = FakeLLM([invalid_output, json.dumps(valid_product_dict)])

    result = generate_product_schema("gold earrings", llm)

    assert result == valid_product_dict
    assert len(llm.prompts) == 2
    repair_prompt = llm.prompts[1]
    assert "gold earrings" in repair_prompt
    assert invalid_output in repair_prompt
    assert "LLM returned invalid JSON" in repair_prompt
    assert "Repair only what is necessary" in repair_prompt


def test_maximum_three_llm_calls():
    llm = FakeLLM(["bad one", "bad two", "bad three"])

    with pytest.raises(GenerationExhaustedError):
        generate_product_schema("gold earrings", llm)

    assert len(llm.prompts) == 3


def test_provider_exception_is_not_retried():
    llm = FakeLLM([LLMProviderError("secret provider detail")])

    with pytest.raises(LLMProviderError, match="secret provider detail"):
        generate_product_schema("gold earrings", llm)

    assert len(llm.prompts) == 1
