import pytest

from tools.domain_tools import (
    CatalogSearchInput,
    MaterialLookupInput,
    PriceEstimateInput,
    estimate_price_range,
    lookup_material,
    search_catalog,
)
from tools.registry import (
    InvalidToolArgumentsError,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    UnknownToolError,
    create_default_registry,
)


def test_material_lookup():
    result = lookup_material(MaterialLookupInput(material="rose gold"))

    assert result["found"] is True
    assert result["material"]["name"] == "gold"
    assert result["material"]["care"]


def test_catalog_search():
    result = search_catalog(CatalogSearchInput(query="silver ring"))

    assert result["match_count"] >= 1
    assert result["products"][0]["id"] == "J-1002"


def test_pricing_tool_is_clearly_an_estimate():
    result = estimate_price_range(
        PriceEstimateInput(category="earrings", materials=["gold"])
    )

    assert result["is_estimate"] is True
    assert result["minimum"] < result["maximum"]
    assert "Estimate only" in result["disclaimer"]


def test_unknown_tool_is_rejected():
    with pytest.raises(UnknownToolError):
        create_default_registry().invoke("run_shell", {})


def test_invalid_tool_arguments_are_rejected():
    with pytest.raises(InvalidToolArgumentsError):
        create_default_registry().invoke("lookup_material", {"material": "   "})


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("search_catalog", {"query": "ring", "limit": 0}),
        ("search_catalog", {"query": "ring", "limit": 6}),
        ("search_knowledge", {"query": "ring care", "top_k": 0}),
        ("search_knowledge", {"query": "ring care", "top_k": 6}),
    ],
)
def test_bounded_integer_tool_arguments_are_rejected(tool_name, arguments):
    with pytest.raises(InvalidToolArgumentsError):
        create_default_registry().invoke(tool_name, arguments)


def test_tool_execution_failure_is_wrapped():
    def failing_tool(arguments):
        raise RuntimeError("internal fixture failure")

    registry = ToolRegistry(
        [
            ToolDefinition(
                name="failing_tool",
                description="A test-only failure.",
                input_model=MaterialLookupInput,
                implementation=failing_tool,
            )
        ]
    )

    with pytest.raises(ToolExecutionError):
        registry.invoke("failing_tool", {"material": "gold"})
