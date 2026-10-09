"""Pydantic schemas for all API request/response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    page_count: int | None = None
    created_at: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int | None = None
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ---------------------------------------------------------------------------
# Search (retrieval without generation)
# ---------------------------------------------------------------------------


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    excerpt: str
    distance: float = Field(description="Lower distance = more similar")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total: int


# ---------------------------------------------------------------------------
# Ask (RAG with generation)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceReference(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    insufficient_evidence: bool
    request_id: str
