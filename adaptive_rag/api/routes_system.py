from typing import Any, Dict

from fastapi import APIRouter, Depends, Response

from ..engine import AdaptiveRAG
from ..observability import API_TRACKER
from .deps import get_jobs, get_rag
from .jobs import SyncJobManager
from .schemas import HealthOut, ReadyOut, SyncJobOut, SyncRequest
from .security import require_api_key

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """Liveness: the process is up. No auth, no dependencies touched."""
    return HealthOut(status="ok")


@router.get("/ready", response_model=ReadyOut)
def ready(
    response: Response,
    rag: AdaptiveRAG = Depends(get_rag),
    jobs: SyncJobManager = Depends(get_jobs),
) -> ReadyOut:
    """Readiness: can this instance answer questions right now? 503 if not.
    Use this for load-balancer / container health checks."""
    info = rag.status()
    ok = bool(info["ready"] and info["qdrant_reachable"])
    if not ok:
        response.status_code = 503
    return ReadyOut(
        ready=ok,
        qdrant_reachable=bool(info["qdrant_reachable"]),
        sync_running=jobs.is_running(),
    )


@router.get("/status", dependencies=[Depends(require_api_key)])
def status(
    rag: AdaptiveRAG = Depends(get_rag),
    jobs: SyncJobManager = Depends(get_jobs),
) -> Dict[str, Any]:
    """Detailed state for an admin page: document counts, failures, last sync."""
    return {**rag.status(), "sync_job": jobs.status()}


@router.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> Dict[str, Any]:
    """Usage counters: LLM calls/tokens (estimated), timings, cache hits."""
    return API_TRACKER.snapshot()


@router.get("/sync", response_model=SyncJobOut, dependencies=[Depends(require_api_key)])
def sync_status(jobs: SyncJobManager = Depends(get_jobs)) -> SyncJobOut:
    return SyncJobOut(**jobs.status())


@router.post(
    "/sync",
    response_model=SyncJobOut,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
def start_sync(
    body: SyncRequest = SyncRequest(),
    jobs: SyncJobManager = Depends(get_jobs),
) -> SyncJobOut:
    """Start (or queue) a sync. force_rebuild wipes and re-embeds everything."""
    return SyncJobOut(**jobs.request(force_rebuild=body.force_rebuild))