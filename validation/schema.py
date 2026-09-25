import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class JewelryProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"]
    style: Literal["minimal", "bold", "vintage", "modern", "boho"]
    materials: list[str]
    colors: list[str]
    price_range: Literal["budget", "mid", "luxury"]
    tags: list[str]

    @field_validator("product_name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("product_name cannot be empty")
        return v.strip()

    @field_validator("materials")
    @classmethod
    def must_have_materials(cls, v):
        if len(v) == 0:
            raise ValueError("At least one material is required")
        return v

    @field_validator("materials", "colors", "tags")
    @classmethod
    def list_items_must_not_be_blank(cls, values):
        cleaned = []
        for value in values:
            if not value.strip():
                raise ValueError("List entries cannot be blank")
            cleaned.append(value.strip())
        return cleaned


def parse_and_validate(raw_json: str) -> JewelryProduct:
    """Parse raw JSON string from LLM and validate it."""
    try:
        data = json.loads(raw_json)
        product = JewelryProduct(**data)
        return product
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")
