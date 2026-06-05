import json
from pipeline.generator_pipeline import generate_product_schema


def run():
    test_cases = [
        "Gold earrings with blue beads, minimal style",
        "Heavy silver biker ring with skull design, bold look",
        "Delicate rose gold necklace with pearl, vintage feel",
    ]

    for test in test_cases:
        print(f"\n📥 Input: {test}")
        try:
            result = generate_product_schema(test)
            print("✅ Valid product schema generated!")
            print(json.dumps(result, indent=2))
        except ValueError as e:
            print(f"❌ Pipeline failed: {e}")
        print("-" * 50)


if __name__ == "__main__":
    run()