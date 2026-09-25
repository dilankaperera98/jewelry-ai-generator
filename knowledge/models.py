from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    source: str
    category: str
    is_synthetic: bool
    content: str


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    source: str
    category: str
    text: str


class RetrievedChunk(KnowledgeChunk):
    score: float = Field(ge=0.0, le=1.0)


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class KnowledgeSearchResult(BaseModel):
    query: str
    result_count: int
    insufficient_context: bool
    results: list[RetrievedChunk]
