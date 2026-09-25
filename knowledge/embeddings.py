import math
import re
from collections import Counter
from typing import Protocol

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by",
    "for", "from", "in", "is", "it", "of", "on", "or", "should", "that",
    "the", "their", "this", "to", "with",
}


class EmbeddingModel(Protocol):
    def fit(self, texts: list[str]) -> None:
        """Learn the embedding vocabulary from indexed documents."""

    def embed(self, text: str) -> list[float]:
        """Convert text to a vector in the fitted embedding space."""


class TfidfEmbeddingModel:
    """Small local TF-IDF embedding model using unigram and bigram features."""

    def __init__(self):
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", text.casefold())
            if token not in STOP_WORDS
        ]
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("At least one text is required to fit embeddings")
        document_frequency = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        features = sorted(document_frequency)
        self._vocabulary = {feature: index for index, feature in enumerate(features)}
        document_count = len(texts)
        self._idf = {
            feature: math.log((1 + document_count) / (1 + frequency)) + 1
            for feature, frequency in document_frequency.items()
        }

    def embed(self, text: str) -> list[float]:
        if not self._vocabulary:
            raise RuntimeError("Embedding model must be fitted before use")
        counts = Counter(self._features(text))
        total = sum(counts.values()) or 1
        vector = [0.0] * len(self._vocabulary)
        for feature, count in counts.items():
            index = self._vocabulary.get(feature)
            if index is not None:
                vector[index] = (count / total) * self._idf[feature]
        return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
