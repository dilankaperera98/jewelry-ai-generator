import os
from anthropic import Anthropic
from dotenv import load_dotenv
from prompts.jewelry_prompt import JEWELRY_SYSTEM_PROMPT

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_llm(user_input: str) -> str:
    """Send user description to Claude, get raw JSON string back."""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=JEWELRY_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    raw = message.content[0].text
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()