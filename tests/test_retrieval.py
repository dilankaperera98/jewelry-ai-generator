import pytest
from pydantic import ValidationError

from knowledge.chunking import chunk_documents
from knowledge.embeddings import TfidfEmbeddingModel
from knowledge.loader import load_documents
from knowledge.models import SearchKnowledgeInput
from knowledge.retriever import InMemoryVectorRetriever, create_default_retriever
from knowledge.tool import search_knowledge
from tools.registry import create_default_registry


def test_document_loading():
    documents = load_documents()

    assert len(documents) == 5
    assert {document.document_id for document in documents} == {
        "DOC-CARE-001",
        "DOC-MFG-001",
        "DOC-CUSTOM-001",
        "DOC-QUALITY-001",
        "DOC-PACK-001",
    }
    assert all(document.is_synthetic for document in documents)


def test_chunk_generation_preserves_metadata_and_overlap():
    document = load_documents()[0]
    chunks = chunk_documents([document], chunk_words=40, overlap_words=10)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "DOC-CARE-001::chunk-001"
    assert chunks[0].document_id == document.document_id
    assert chunks[0].title == document.title
    assert chunks[0].source == document.source
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_embedding_and_index_creation():
    chunks = chunk_documents(load_documents())
    model = TfidfEmbeddingModel()
    retriever = InMemoryVectorRetriever(chunks, model)

    vector = model.embed("ceramic firing shrinkage")
    assert retriever.chunk_count == len(chunks)
    assert len(vector) > 0
    assert any(value > 0 for value in vector)


def test_relevant_retrieval():
    results = create_default_retriever().retrieve(
        "How should pearl jewelry be cleaned and stored?", top_k=3
    )

    assert results[0].document_id == "DOC-CARE-001"
    assert results[0].score > 0


def test_top_k_is_bounded_and_enforced():
    retriever = create_default_retriever()

    assert len(retriever.retrieve("ceramic jewelry", top_k=2)) <= 2
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("ceramic jewelry", top_k=6)
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(query="ceramic jewelry", top_k=6)


def test_empty_query_is_rejected():
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(query="   ")
    with pytest.raises(ValueError, match="empty"):
        create_default_retriever().retrieve("   ")


def test_retrieval_preserves_source_metadata():
    result = create_default_retriever().retrieve("engraving character limit", 1)[0]

    assert result.document_id == "DOC-CUSTOM-001"
    assert result.chunk_id.startswith("DOC-CUSTOM-001::")
    assert result.title
    assert result.source == "synthetic_demo_customization_guide"


def test_local_search_knowledge_tool():
    result = create_default_registry().invoke(
        "search_knowledge",
        {"query": "protect fragile ceramic jewelry during shipping", "top_k": 2},
    )

    assert result["insufficient_context"] is False
    assert result["result_count"] >= 1
    assert result["results"][0]["document_id"] == "DOC-PACK-001"


def test_insufficient_context_is_explicit():
    result = search_knowledge(
        SearchKnowledgeInput(query="quantum rocket propulsion telemetry", top_k=3)
    )

    assert result["insufficient_context"] is True
    assert result["results"] == []
