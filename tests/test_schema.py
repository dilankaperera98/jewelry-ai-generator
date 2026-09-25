import json

import pytest

from validation.schema import parse_and_validate


def test_schema_success(valid_product_dict):
    product = parse_and_validate(json.dumps(valid_product_dict))

    assert product.model_dump() == valid_product_dict


def test_malformed_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_and_validate("{not-json}")


def test_invalid_enum(valid_product_dict):
    valid_product_dict["category"] = "brooch"

    with pytest.raises(ValueError, match="Validation failed"):
        parse_and_validate(json.dumps(valid_product_dict))


def test_unexpected_fields(valid_product_dict):
    valid_product_dict["internal_note"] = "must not be accepted"

    with pytest.raises(ValueError, match="extra_forbidden"):
        parse_and_validate(json.dumps(valid_product_dict))


def test_blank_material_entries(valid_product_dict):
    valid_product_dict["materials"] = ["gold", "   "]

    with pytest.raises(ValueError, match="List entries cannot be blank"):
        parse_and_validate(json.dumps(valid_product_dict))
