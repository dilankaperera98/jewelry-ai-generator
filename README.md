# Jewelry AI Generator - Structured LLM Pipeline & API

A modular, production-inspired AI pipeline that converts natural language jewelry descriptions into validated, structured JSON schemas - exposed as a REST API.

Built to explore the challenges of making LLM outputs reliable, consistent, and usable in real-world backend systems.

## Problem

LLMs are powerful but non-deterministic, they can return inconsistent formats, ignore instructions, or produce malformed outputs. Downstream systems (databases, product catalogs, rendering engines) need structured, validated data.

This project addresses that gap.


## Solution

A self-correcting pipeline that:
1. Takes plain English input
2. Sends it to an LLM with structured prompting
3. Validates the output against a strict schema
4. Retries with refined prompts on failure
5. Returns clean, production-ready JSON via a REST API


## Architecture

User Input

Prompt Builder (prompts/)
    ↓
LLM Engine - Claude (core/)
    ↓
Retry Loop (pipeline/)
    →Validate 
    → Refine prompt (on failure)
    ↓
Pydantic Validation (validation/)
    ↓
FastAPI Endpoint (api/)
    ↓
Structured JSON Output

## Project Structure

jewelry-ai-generator/
├── core/                  # LLM engine — Claude API integration
├── prompts/               # Prompt templates — separated for easy iteration
├── validation/            # Pydantic schema enforcement
├── pipeline/              # Orchestration + self-correcting retry loop
├── api/                   # FastAPI routes and request/response models
├── rag/                   # Retrieval layer (in development)
├── server.py              # API server entry point
└── main.py                # CLI entry point

## Key Engineering Decisions

Separation of concerns - LLM logic, prompt templates, validation, and orchestration are fully decoupled. Each layer can be modified or replaced independently.

Self-correcting retry loop - on validation failure, the pipeline automatically retries with progressively stricter prompts rather than failing immediately. This handles LLM non-determinism gracefully.

Schema-first validation — Pydantic enforces strict type checking and field constraints before any output is returned. Invalid LLM responses are caught and handled, never passed downstream.

Model-agnostic design — the LLM engine is isolated in `core/`, making it straightforward to swap Claude for GPT-4, Gemini, or any other provider.

## API

### Run the server
```bash
python3 server.py
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/generate` | Generate schema from description |

### Example Request
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Gold earrings with blue beads, minimal style"}'
```

### Example Response
```json
{
  "success": true,
  "input": "Gold earrings with blue beads, minimal style",
  "schema": {
    "product_name": "Gold Minimal Blue Bead Earrings",
    "category": "earrings",
    "style": "minimal",
    "materials": ["gold", "beads"],
    "colors": ["gold", "blue"],
    "price_range": "mid",
    "tags": ["gold earrings", "blue beads", "minimal", "everyday wear"]
  }
}
```

### Interactive Docs
Visit `http://localhost:8000/docs` for auto-generated Swagger UI.

## Tech Stack

- Python 3.14+
- Anthropic Claude API
- Pydantic (schema validation)
- FastAPI (REST API)
- Uvicorn (ASGI server)
- python-dotenv

---

## Setup

```bash
git clone https://github.com/dilankaperera98/jewelry-ai-generator
cd jewelry-ai-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:
ANTHROPIC_API_KEY=your-key-here

Run CLI:
```bash
python3 main.py
```

Run API:
```bash
python3 server.py
```

## Future Development

- RAG layer — retrieve similar past designs to enrich prompt context before generation
- Agent orchestration — multi-step reasoning for complex custom design requests
- Image generation - extend pipeline to generate visual previews via DALL-E
- Human-in-the-loop — review and override interface for generated schemas
- Frontend — React-based design configurator for end users

## Context

This project applies backend reliability thinking - separation of concerns, validation layers, retry logic, and structured outputs to the challenges of LLM integration. The same engineering principles used in high-throughput financial systems apply directly to making AI pipelines production-ready.