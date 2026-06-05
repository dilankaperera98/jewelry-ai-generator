from core.llm_engine import call_llm
from validation.schema import parse_and_validate


def refine_prompt(original_input: str, attempt: int) -> str:
    """Make the prompt stricter on retry attempts."""
    if attempt == 1:
        return f"{original_input}. Respond with raw JSON only, no extra text."
    return f"{original_input}. IMPORTANT: Return ONLY a JSON object, nothing else."


def generate_product_schema(user_input: str) -> dict:
    """
    Full pipeline with retry loop:
    Input → LLM → Validate → Retry if needed → Output
    """
    last_error = None

    for attempt in range(3):
        try:
            print(f"  ⏳ Attempt {attempt + 1}/3...")

            # Refine prompt on retries
            prompt = refine_prompt(user_input, attempt)

            # Step 1: Call LLM
            raw_json = call_llm(prompt)

            # Step 2: Validate
            product = parse_and_validate(raw_json)

            # Step 3: Return clean dict
            return product.model_dump()

        except ValueError as e:
            last_error = e
            print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
            continue

    raise ValueError(f"Pipeline failed after 3 attempts. Last error: {last_error}")