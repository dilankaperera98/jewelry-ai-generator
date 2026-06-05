from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.generator_pipeline import generate_product_schema

app = FastAPI(
    title="Jewelry AI Generator",
    description="LLM pipeline that converts jewelry descriptions into validated JSON schemas",
    version="1.0.0"
)


class JewelryRequest(BaseModel):
    description: str


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "Jewelry AI Generator is running"}


@app.post("/generate")
def generate(request: JewelryRequest):
    """
    Convert a jewelry description into a validated JSON schema.

    Example input:
    {
        "description": "Gold earrings with blue beads, minimal style"
    }
    """
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    try:
        result = generate_product_schema(request.description)
        return {
            "success": True,
            "input": request.description,
            "schema": result
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))