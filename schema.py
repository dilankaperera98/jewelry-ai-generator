from pydantic import BaseModel, field_validator
from typing import List, Literal
import json


class JewelryProduct(BaseModel):
    product_name: str
    category: Literal["earrings", "necklace", "bracelet", "ring", "other"]
    style: Literal["minimal", "bold", "vintage", "modern", "boho"]
    materials: List[str]
    colors: List[str]
    price_range: Literal["budget", "mid", "luxury"]
    tags: List[str]

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