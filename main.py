import json

from core.config import load_settings
from core.llm_engine import AnthropicLLMClient
from pipeline.generator_pipeline import generate_product_schema


def run():
    llm_client = AnthropicLLMClient(load_settings())
    test_cases = [
        "Gold earrings with blue beads, minimal style",
        "Ceramic Bracelets with red and white beads",
    ]

    for test in test_cases:
        print(f"\n📥 Input: {test}")
        try:
            result = generate_product_schema(test, llm_client)
            print("✅ Valid product schema generated!")
            print(json.dumps(result, indent=2))
        except ValueError as e:
            print(f"❌ Pipeline failed: {e}")
        print("-" * 50)


if __name__ == "__main__":
    run()
