from typing import Annotated, Literal

from mcp.server import MCPServer
from pydantic import BaseModel, Field

from knowledge.models import KnowledgeSearchResult, SearchKnowledgeInput
from knowledge.tool import search_knowledge as search_knowledge_domain
from tools.domain_tools import (
    CatalogSearchInput,
    MaterialLookupInput,
    PriceEstimateInput,
    estimate_price_range as estimate_price_range_domain,
    lookup_material as lookup_material_domain,
    search_catalog as search_catalog_domain,
)

mcp_server = MCPServer(
    "jewelry-domain-tools",
    description="Approved deterministic jewelry tools and curated knowledge retrieval.",
)


class MaterialInfo(BaseModel):
    name: str
    aliases: list[str]
    durability: str
    care: str
    hypoallergenic: bool
    price_tier: Literal["budget", "mid", "luxury"]


class MaterialLookupOutput(BaseModel):
    found: bool
    material: MaterialInfo | None = None
    requested_material: str | None = None
    supported_materials: list[str] | None = None


class CatalogProduct(BaseModel):
    id: str
    name: str
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"]
    style: Literal["minimal", "bold", "vintage", "modern", "boho"]
    materials: list[str]
    colors: list[str]
    price_usd: int


class CatalogSearchOutput(BaseModel):
    query: str
    match_count: int
    products: list[CatalogProduct]


class PriceEstimateOutput(BaseModel):
    is_estimate: Literal[True]
    currency: Literal["USD"]
    minimum: int
    maximum: int
    basis: str
    disclaimer: str


@mcp_server.tool(
    name="lookup_material",
    description="Look up supported jewelry material properties and care guidance from local data.",
    structured_output=True,
)
def lookup_material(
    material: Annotated[str, Field(min_length=1)],
) -> MaterialLookupOutput:
    """Look up one material or material alias."""
    arguments = MaterialLookupInput(material=material)
    return MaterialLookupOutput.model_validate(lookup_material_domain(arguments))


@mcp_server.tool(
    name="search_catalog",
    description="Search the local jewelry catalog with deterministic matching.",
    structured_output=True,
)
def search_catalog(
    query: Annotated[str, Field(min_length=1)],
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"] | None = None,
    limit: Annotated[int, Field(ge=1, le=5)] = 3,
) -> CatalogSearchOutput:
    """Search local products by text and optional category."""
    arguments = CatalogSearchInput(query=query, category=category, limit=limit)
    return CatalogSearchOutput.model_validate(search_catalog_domain(arguments))


@mcp_server.tool(
    name="estimate_price_range",
    description="Return a clearly labelled USD estimate using local catalog and material pricing data.",
    structured_output=True,
)
def estimate_price_range(
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"],
    materials: Annotated[list[str], Field(min_length=1)],
) -> PriceEstimateOutput:
    """Estimate a local price range; this is not a quote."""
    arguments = PriceEstimateInput(category=category, materials=materials)
    return PriceEstimateOutput.model_validate(estimate_price_range_domain(arguments))


@mcp_server.tool(
    name="search_knowledge",
    description="Search curated unstructured jewelry care, manufacturing, customization, quality, and packaging guidance.",
    structured_output=True,
)
def search_knowledge(
    query: Annotated[str, Field(min_length=1)],
    top_k: Annotated[int, Field(ge=1, le=5)] = 3,
) -> KnowledgeSearchResult:
    """Retrieve grounded chunks; exact business data belongs in deterministic tools."""
    arguments = SearchKnowledgeInput(query=query, top_k=top_k)
    return KnowledgeSearchResult.model_validate(search_knowledge_domain(arguments))


def main() -> None:
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
