# 💍 Jewelry AI Generator

An AI-powered pipeline that converts natural language jewelry descriptions into validated, structured JSON schemas.

## What it does
- Takes plain English input like "Gold earrings with blue beads, minimal style"
- Sends it to Claude (Anthropic LLM)
- Converts it into structured JSON
- Validates the output with Pydantic
- Returns a clean, production-ready schema

## Tech Stack
- Python 3.14+
- Anthropic Claude API
- Pydantic (validation)
- python-dotenv

## Example
Input:
"Gold earrings with blue beads, minimal style"

Output:
{
  "product_name": "Gold Minimal Blue Bead Earrings",
  "category": "earrings",
  "style": "minimal",
  "materials": ["gold", "beads"],
  "colors": ["gold", "blue"],
  "price_range": "mid",
  "tags": ["gold earrings", "blue beads", "minimal", "everyday wear"]
}

## Setup
1. Clone the repo
2. Create virtual environment: python3 -m venv venv
3. Activate it: source venv/bin/activate
4. Install dependencies: pip install anthropic pydantic python-dotenv
5. Add your API key to .env: ANTHROPIC_API_KEY=your-key-here
6. Run: python3 main.py