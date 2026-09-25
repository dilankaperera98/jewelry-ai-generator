import json

from fastapi.testclient import TestClient

from api.routes import app, get_llm_client
from core.llm_engine import LLMProviderError


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompt):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def set_fake_llm(fake):
    app.dependency_overrides[get_llm_client] = lambda: fake


def teardown_function():
    app.dependency_overrides.clear()


def test_generate_success(valid_product_dict):
    set_fake_llm(FakeLLM([json.dumps(valid_product_dict)]))

    response = TestClient(app).post(
        "/generate", json={"description": "gold earrings"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "input": "gold earrings",
        "schema": valid_product_dict,
    }


def test_blank_input():
    set_fake_llm(FakeLLM([]))

    response = TestClient(app).post("/generate", json={"description": "   "})

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "invalid_input",
            "message": "Description cannot be empty",
        }
    }


def test_validation_exhaustion_does_not_leak_validation_details():
    set_fake_llm(FakeLLM(["secret invalid output"] * 3))

    response = TestClient(app).post(
        "/generate", json={"description": "gold earrings"}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "generation_failed",
            "message": "Unable to generate a valid product schema",
        }
    }
    assert "secret" not in response.text


def test_provider_failure_does_not_leak_provider_details():
    set_fake_llm(FakeLLM([LLMProviderError("secret upstream detail")]))

    response = TestClient(app).post(
        "/generate", json={"description": "gold earrings"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "provider_unavailable",
            "message": "The generation service is temporarily unavailable",
        }
    }
    assert "secret upstream detail" not in response.text
