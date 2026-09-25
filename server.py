import uvicorn

from observability.logging import configure_structured_logging

if __name__ == "__main__":
    configure_structured_logging()
    uvicorn.run(
        "api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
