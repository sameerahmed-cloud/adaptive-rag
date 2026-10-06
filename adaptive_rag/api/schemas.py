from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..config import MAX_QUESTION_CHARS

RagMode = Literal["auto", "semantic", "keyword", "hybrid", "summary"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: RagMode = "auto"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class SourceOut(BaseModel):
    context: Optional[int] = None      # matches the [Context N] marker in the answer
    file_name: Optional[str] = None
    relative_path: Optional[str] = None
    document_id: Optional[str] = None
    sheet_name: Optional[str] = None
    header_path: Optional[str] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    snippet: str = ""


class EvaluationOut(BaseModel):
    relevant: bool
    faithful: bool
    complete: bool
    verdict: str
    issues: List[str] = []


class AskResponse(BaseModel):
    answer: str
    status: Literal["verified", "no_evidence", "unverified"]
    verified: bool
    strategy: str
    strategy_reason: str
    attempts: int
    sources: List[SourceOut]
    evaluation: Optional[EvaluationOut] = None
    latency_seconds: float
    request_id: str


class DocumentOut(BaseModel):
    path: str
    status: Literal["indexed", "failed"]
    extension: Optional[str] = None
    size_bytes: Optional[int] = None
    indexed_nodes: int = 0
    error: Optional[str] = None        # also set when the LATEST edit failed but an older version is still served
    attempts: Optional[int] = None


class SyncRequest(BaseModel):
    force_rebuild: bool = False


class SyncJobOut(BaseModel):
    state: Literal["idle", "running"]
    queued: bool                        # another sync is waiting to start after this one
    force_rebuild: bool
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_report: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None


class UploadResponse(BaseModel):
    path: str
    size_bytes: int
    replaced: bool
    sync: Optional[SyncJobOut] = None


class HealthOut(BaseModel):
    status: str


class ReadyOut(BaseModel):
    ready: bool
    qdrant_reachable: bool
    sync_running: bool