import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv() #open env file

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_jewelry_json(user_input: str) -> str:
    """Send user description to Claude, get raw JSON string back."""

    system_prompt = """You are a jewelry product data assistant.
When given a jewelry description, respond with ONLY a valid JSON object.
No explanation, no markdown, no code blocks — raw JSON only.

The JSON must follow this exact structure:
{
  "product_name": "string",
  "category": "earrings | necklace | bracelet | ring | other",
  "style": "minimal | bold | vintage | modern | boho",
  "materials": ["list", "of", "materials"],
  "colors": ["list", "of", "colors"],
  "price_range": "budget | mid | luxury",
  "tags": ["list", "of", "search", "tags"]
}"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    raw = message.content[0].text
    # Strip markdown code blocks if Claude adds them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # remove first line (```json)
        raw = raw.rsplit("```", 1)[0]  # remove last ```
    return raw.strip()


# Quick test
if __name__ == "__main__":
    test_input = "Gold earrings with blue beads, minimal style"
    raw = get_jewelry_json(test_input)
    print("Raw LLM output:")
    print(raw)