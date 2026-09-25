import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DATA_DIRECTORY = Path(__file__).parent / "data"


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaterialLookupInput(ToolInput):
    material: str = Field(min_length=1)

    @field_validator("material")
    @classmethod
    def material_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("material cannot be blank")
        return value.strip()


class CatalogSearchInput(ToolInput):
    query: str = Field(min_length=1)
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"] | None = None
    limit: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class PriceEstimateInput(ToolInput):
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"]
    materials: list[str] = Field(min_length=1)

    @field_validator("materials")
    @classmethod
    def materials_must_not_contain_blanks(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip().lower() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("materials cannot contain blank entries")
        return cleaned


@lru_cache
def _load_data(filename: str) -> list[dict]:
    with (DATA_DIRECTORY / filename).open(encoding="utf-8") as data_file:
        return json.load(data_file)


def lookup_material(arguments: MaterialLookupInput) -> dict:
    requested = arguments.material.casefold()
    for material in _load_data("materials.json"):
        names = [material["name"], *material["aliases"]]
        if requested in {name.casefold() for name in names}:
            return {"found": True, "material": material}
    return {
        "found": False,
        "requested_material": arguments.material,
        "supported_materials": [item["name"] for item in _load_data("materials.json")],
    }


def search_catalog(arguments: CatalogSearchInput) -> dict:
    query_tokens = set(re.findall(r"[a-z0-9]+", arguments.query.casefold()))
    matches = []
    for product in _load_data("catalog.json"):
        if arguments.category and product["category"] != arguments.category:
            continue
        searchable = " ".join(
            [
                product["name"],
                product["category"],
                product["style"],
                *product["materials"],
                *product["colors"],
            ]
        ).casefold()
        score = sum(token in searchable for token in query_tokens)
        if score:
            matches.append((score, product))
    matches.sort(key=lambda item: (-item[0], item[1]["id"]))
    products = [product for _, product in matches[: arguments.limit]]
    return {"query": arguments.query, "match_count": len(products), "products": products}


def estimate_price_range(arguments: PriceEstimateInput) -> dict:
    requested_materials = set(arguments.materials)
    category_products = [
        product
        for product in _load_data("catalog.json")
        if arguments.category == "other" or product["category"] == arguments.category
    ]
    close_matches = [
        product
        for product in category_products
        if requested_materials.intersection(product["materials"])
    ]
    basis = close_matches or category_products

    if basis:
        observed_prices = [product["price_usd"] for product in basis]
        minimum = round(min(observed_prices) * 0.85)
        maximum = round(max(observed_prices) * 1.15)
    else:
        tiers = {
            item["name"]: item["price_tier"] for item in _load_data("materials.json")
        }
        tier_ranges = {"budget": (30, 90), "mid": (80, 250), "luxury": (225, 1000)}
        tier_rank = {"budget": 0, "mid": 1, "luxury": 2}
        selected_tiers = [tiers[name] for name in arguments.materials if name in tiers]
        highest_tier = max(selected_tiers, key=tier_rank.get) if selected_tiers else None
        selected_range = tier_ranges[highest_tier] if highest_tier else (50, 300)
        minimum, maximum = selected_range

    return {
        "is_estimate": True,
        "currency": "USD",
        "minimum": minimum,
        "maximum": maximum,
        "basis": f"{len(basis)} local catalog product(s)",
        "disclaimer": "Estimate only; final pricing requires a detailed design and current supplier costs.",
    }
