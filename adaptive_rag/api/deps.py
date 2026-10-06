from contextlib import contextmanager

from fastapi import HTTPException, Request

from ..config import QUERY_QUEUE_TIMEOUT_SECONDS
from ..engine import AdaptiveRAG
from .jobs import SyncJobManager


def get_rag(request: Request) -> AdaptiveRAG:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="Service is starting up.")
    return rag


def get_jobs(request: Request) -> SyncJobManager:
    return request.app.state.jobs


@contextmanager
def query_slot(request: Request):
    """Limit how many questions run at once. Every question costs at least two
    Gemini calls, so unlimited concurrency would burn quota and cause the
    timeouts you saw earlier."""
    slots = request.app.state.query_slots
    if not slots.acquire(timeout=QUERY_QUEUE_TIMEOUT_SECONDS):
        raise HTTPException(
            status_code=429,
            detail="The system is busy answering other questions. Try again shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        slots.release()