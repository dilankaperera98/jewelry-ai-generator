import pytest


@pytest.fixture
def valid_product_dict():
    return {
        "product_name": "Gold Minimal Earrings",
        "category": "earrings",
        "style": "minimal",
        "materials": ["gold"],
        "colors": ["gold"],
        "price_range": "mid",
        "tags": ["gold earrings", "minimal"],
    }
