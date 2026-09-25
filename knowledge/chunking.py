from knowledge.models import KnowledgeChunk, KnowledgeDocument

DEFAULT_CHUNK_WORDS = 80
DEFAULT_OVERLAP_WORDS = 15


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[KnowledgeChunk]:
    """Split documents into transparent word windows with a small context overlap.

    Eighty words keeps demo chunks focused and inspectable. Fifteen overlapping words
    reduce the chance that a relevant phrase is split exactly at a chunk boundary.
    """
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Chunk size must be positive and overlap smaller than the chunk")

    chunks = []
    step = chunk_words - overlap_words
    for document in documents:
        words = document.content.split()
        for index, start in enumerate(range(0, len(words), step), start=1):
            text = " ".join(words[start : start + chunk_words])
            if not text:
                continue
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.document_id}::chunk-{index:03d}",
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    category=document.category,
                    text=text,
                )
            )
            if start + chunk_words >= len(words):
                break
    return chunks
