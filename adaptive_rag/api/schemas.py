from typing import Literal

from pydantic import BaseModel, Field


RAGMode = Literal["auto", "semantic", "keyword", "hybrid", "summary"]


class HealthResponse(BaseModel):
    status: str
    knowledge_base: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    rag_mode: RAGMode = "auto"
    max_retries: int = Field(default=3, ge=1, le=3)


class QueryResponse(BaseModel):
    answer: str


class SyncResponse(BaseModel):
    message: str


class DocumentResponse(BaseModel):
    filename: str
    document_id: str | None = None
    size: int | None = None
    extension: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    message: str
