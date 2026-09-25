from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from knowledge.models import SearchKnowledgeInput
from knowledge.tool import search_knowledge
from tools.domain_tools import (
    CatalogSearchInput,
    MaterialLookupInput,
    PriceEstimateInput,
    estimate_price_range,
    lookup_material,
    search_catalog,
)


class ToolRegistryError(RuntimeError):
    """Base error for safe registry failures."""


class UnknownToolError(ToolRegistryError):
    """Raised when a model requests a tool outside the registry."""


class InvalidToolArgumentsError(ToolRegistryError):
    """Raised when model-supplied arguments fail validation."""


class ToolExecutionError(ToolRegistryError):
    """Raised when an approved deterministic tool fails."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    implementation: Callable[[BaseModel], dict]

    def public_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]):
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique")

    def definitions(self) -> list[dict]:
        return [tool.public_schema() for tool in self._tools.values()]

    def validate_arguments(self, name: str, raw_arguments: dict) -> BaseModel:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {name}")
        try:
            return tool.input_model.model_validate(raw_arguments)
        except ValidationError as exc:
            raise InvalidToolArgumentsError(f"Invalid arguments for tool: {name}") from exc

    def invoke(self, name: str, raw_arguments: dict) -> dict:
        tool = self._tools.get(name)
        arguments = self.validate_arguments(name, raw_arguments)
        try:
            return tool.implementation(arguments)
        except ToolRegistryError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool execution failed: {name}") from exc


def create_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="lookup_material",
                description="Look up supported jewelry material properties and care guidance from local data.",
                input_model=MaterialLookupInput,
                implementation=lookup_material,
            ),
            ToolDefinition(
                name="search_catalog",
                description="Search the local jewelry catalog using deterministic text and optional category matching.",
                input_model=CatalogSearchInput,
                implementation=search_catalog,
            ),
            ToolDefinition(
                name="estimate_price_range",
                description="Estimate a USD price range from local catalog and material pricing data. The result is explicitly an estimate.",
                input_model=PriceEstimateInput,
                implementation=estimate_price_range,
            ),
            ToolDefinition(
                name="search_knowledge",
                description="Search curated unstructured jewelry care, manufacturing, customization, quality, and packaging guidance. Use deterministic tools for exact materials, catalog products, or prices.",
                input_model=SearchKnowledgeInput,
                implementation=search_knowledge,
            ),
        ]
    )
