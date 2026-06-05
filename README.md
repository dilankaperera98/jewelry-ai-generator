# Jewelry AI Generator

A Python-based AI pipeline that converts unstructured natural language inputs into validated, structured JSON schemas using LLMs.

## Overview

This project demonstrates how large language models can be integrated into backend systems to transform ambiguous user inputs into reliable, production-ready data.

The focus is not just on generating outputs, but on ensuring consistency, validation, and reliability through a structured pipeline.

## Architecture

User Input
-> LLM (Anthropic Claude)
-> Raw JSON Output
-> Validation Layer (Pydantic)
-> Clean Structured Schema

## Key Features

- Structured Output Generation**: Converts free-form text into predefined JSON schema using prompt engineering
- Validation Layer**: Uses Pydantic to enforce schema correctness and prevent malformed outputs
- Error Handling**: Handles invalid or inconsistent LLM responses (extendable to retry/self-correction loops)
- Pipeline-Oriented Design**: Separates generation, validation, and transformation steps for reliability

## Example

Input
"Gold earrings with blue beads, minimal style"

Output

{
  "product_name": "Gold Minimal Blue Bead Earrings",
  "category": "earrings",
  "style": "minimal",
  "materials": ["gold", "beads"]
  "colors": ["gold", "blue"],
  "price_range": "mid",
  "tags": ["gold earrings", "blue beads", "minimal", "everyday wear"]
}

## Tech Stack

- Python
- Anthropic Claude API
- Pydantic (schema validation)
- python-dotenv

## Design Considerations

- LLM outputs are non-deterministic → addressed via schema validation
- Downstream systems require structured data → enforced through JSON schema
- Reliability over raw generation → focus on making outputs usable in production pipelines

## Future Improvements

- Add retry/self-correction loop for invalid outputs
- Extend pipeline to generate visual previews (image generation layer)
- Introduce RAG-based enrichment for contextual recommendations
- Wrap as a FastAPI service for production deployment

## Purpose

This project is part of my work in building AI-driven systems that bridge natural language interfaces with structured backend workflows, with a focus on reliability and real-world applicability.
