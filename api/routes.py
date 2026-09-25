from functools import lru_cache
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import ConfigurationError, load_settings
from core.llm_engine import (
    AnthropicLLMClient,
    LLMClient,
    LLMProviderError,
    ToolUsingLLMClient,
)
from mcp_integration.client import MCPClientAdapter
from mcp_integration.executor import MCPToolExecutor
from observability.tracing import RunTrace
from pipeline.agent_orchestrator import (
    AgentOrchestrationError,
    AgentResult,
    run_agent,
)
from pipeline.generator_pipeline import (
    GenerationExhaustedError,
    generate_product_schema,
)
from validation.schema import JewelryProduct
from tools.executor import LocalToolExecutor, ToolExecutor
from tools.registry import ToolRegistry, create_default_registry

app = FastAPI(
    title="Jewelry AI Generator",
    description="LLM pipeline that converts jewelry descriptions into validated JSON schemas",
    version="1.0.0"
)


class JewelryRequest(BaseModel):
    description: str


class AgentJewelryRequest(JewelryRequest):
    tool_execution: Literal["local", "mcp"] = "local"


class GenerateResponse(BaseModel):
    success: bool
    input: str
    schema_: JewelryProduct = Field(alias="schema")

    model_config = {"populate_by_name": True}


class AgentGenerateResponse(AgentResult):
    success: bool
    input: str
    run_id: str


class HealthResponse(BaseModel):
    status: str
    message: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


@lru_cache
def get_llm_client() -> LLMClient:
    """Create the production LLM client lazily from application settings."""
    return AnthropicLLMClient(load_settings())


@lru_cache
def get_agent_llm_client() -> ToolUsingLLMClient:
    """Create the production tool-capable client lazily."""
    return AnthropicLLMClient(load_settings())


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return create_default_registry()


@lru_cache
def get_local_tool_executor() -> LocalToolExecutor:
    return LocalToolExecutor(get_tool_registry())


@lru_cache
def get_mcp_tool_executor() -> MCPToolExecutor:
    return MCPToolExecutor(get_tool_registry(), MCPClientAdapter())


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return error_response(400, "invalid_request", "The request body is invalid")


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(
    request: Request, exc: ConfigurationError
) -> JSONResponse:
    return error_response(
        503,
        "provider_unavailable",
        "The generation service is temporarily unavailable",
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "Jewelry AI Generator is running"}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    response_model_by_alias=True,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Generation exhausted"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
def generate(
    request: JewelryRequest, llm_client: LLMClient = Depends(get_llm_client)
):
    """
    Convert a jewelry description into a validated JSON schema.

    Example input:
    {
        "description": "Gold earrings with blue beads, minimal style"
    }
    """
    if not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_input", "message": "Description cannot be empty"},
        )

    try:
        run_id = str(uuid4())
        result = generate_product_schema(
            request.description,
            llm_client,
            RunTrace(run_id, "generation"),
        )
        return {
            "success": True,
            "input": request.description,
            "schema": result,
        }
    except GenerationExhaustedError:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "generation_failed",
                "message": "Unable to generate a valid product schema",
            },
        )
    except (LLMProviderError, ConfigurationError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_unavailable",
                "message": "The generation service is temporarily unavailable",
            },
        )


@app.post(
    "/agent/generate",
    response_model=AgentGenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Agent run failed"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
def agent_generate(
    request: AgentJewelryRequest,
    llm_client: ToolUsingLLMClient = Depends(get_agent_llm_client),
    local_executor: LocalToolExecutor = Depends(get_local_tool_executor),
    mcp_executor: MCPToolExecutor = Depends(get_mcp_tool_executor),
):
    """Generate a proposal through bounded native tool-use orchestration."""
    if not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_input", "message": "Description cannot be empty"},
        )

    run_id = str(uuid4())
    executors: dict[str, ToolExecutor] = {
        "local": local_executor,
        "mcp": mcp_executor,
    }
    try:
        result = run_agent(
            request.description,
            llm_client,
            executors[request.tool_execution],
            run_id,
        )
        return {
            "success": True,
            "input": request.description,
            "run_id": run_id,
            **result.model_dump(),
        }
    except AgentOrchestrationError:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "agent_failed",
                "message": "Unable to complete the bounded generation workflow",
            },
        )
    except (LLMProviderError, ConfigurationError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_unavailable",
                "message": "The generation service is temporarily unavailable",
            },
        )
