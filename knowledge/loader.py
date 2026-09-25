import json
from pathlib import Path

from knowledge.models import KnowledgeDocument

DEFAULT_DOCUMENT_PATH = Path(__file__).parent / "data" / "documents_v1.json"


def load_documents(path: Path = DEFAULT_DOCUMENT_PATH) -> list[KnowledgeDocument]:
    """Load and validate a versioned local knowledge document collection."""
    with path.open(encoding="utf-8") as document_file:
        payload = json.load(document_file)
    if payload.get("version") != "1.0":
        raise ValueError("Unsupported knowledge-base version")
    return [KnowledgeDocument.model_validate(item) for item in payload["documents"]]
