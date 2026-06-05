import json
from llm_client import get_jewelry_json
from schema import parse_and_validate


def generate_product_schema(user_input: str) -> dict:
    """Full pipeline: input → LLM → validate → clean JSON output."""

    print(f"\n📥 Input: {user_input}")
    print("⏳ Calling Claude...")

    # Step 1: Get raw JSON from LLM
    raw_json = get_jewelry_json(user_input)

    # Step 2: Validate with Pydantic
    product = parse_and_validate(raw_json)

    # Step 3: Return as clean dict
    result = product.model_dump()
    print("✅ Valid product schema generated!")
    return result


if __name__ == "__main__":
    test_cases = [
        "Gold earrings with blue beads, minimal style",
        "Heavy silver biker ring with skull design, bold look",
        "Delicate rose gold necklace with pearl, vintage feel",
    ]

    for test in test_cases:
        try:
            result = generate_product_schema(test)
            print(json.dumps(result, indent=2))
        except ValueError as e:
            print(f"❌ Error: {e}")
        print("-" * 50)