from knowledge.models import KnowledgeSearchResult, SearchKnowledgeInput
from knowledge.retriever import Retriever, create_default_retriever


def search_knowledge(
    arguments: SearchKnowledgeInput,
    retriever: Retriever | None = None,
) -> dict:
    """Search only the curated local knowledge collection."""
    active_retriever = retriever or create_default_retriever()
    results = active_retriever.retrieve(arguments.query, arguments.top_k)
    return KnowledgeSearchResult(
        query=arguments.query,
        result_count=len(results),
        insufficient_context=not results,
        results=results,
    ).model_dump()
