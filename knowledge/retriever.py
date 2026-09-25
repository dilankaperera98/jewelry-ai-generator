from functools import lru_cache
from typing import Protocol

from knowledge.chunking import chunk_documents
from knowledge.embeddings import EmbeddingModel, TfidfEmbeddingModel, cosine_similarity
from knowledge.loader import load_documents
from knowledge.models import KnowledgeChunk, RetrievedChunk

MAX_TOP_K = 5
DEFAULT_MINIMUM_SCORE = 0.08


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return bounded chunks ranked by descending relevance."""


class InMemoryVectorRetriever:
    """Tiny in-memory cosine index suitable for the inspectable demo corpus."""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        embedding_model: EmbeddingModel | None = None,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ):
        if not chunks:
            raise ValueError("At least one chunk is required")
        self._chunks = chunks
        self._embedding_model = embedding_model or TfidfEmbeddingModel()
        self._embedding_model.fit([chunk.text for chunk in chunks])
        self._vectors = [self._embedding_model.embed(chunk.text) for chunk in chunks]
        self._minimum_score = minimum_score

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        query_vector = self._embedding_model.embed(query)
        ranked = sorted(
            (
                (cosine_similarity(query_vector, vector), chunk)
                for chunk, vector in zip(self._chunks, self._vectors)
            ),
            key=lambda item: (-item[0], item[1].chunk_id),
        )
        return [
            RetrievedChunk(**chunk.model_dump(), score=round(score, 6))
            for score, chunk in ranked[:top_k]
            if score >= self._minimum_score
        ]


@lru_cache
def create_default_retriever() -> InMemoryVectorRetriever:
    return InMemoryVectorRetriever(chunk_documents(load_documents()))
