import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import API_KEYS, CORS_ORIGINS, MAX_CONCURRENT_QUERIES, SYNC_ON_STARTUP
from ..engine import AdaptiveRAG
from . import routes_documents, routes_query, routes_system
from .jobs import SyncJobManager

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Show this package's INFO logs (chunker profiling, request log) without
    # enabling every library's chatter.
    package = __name__.split(".")[0]
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
    logging.getLogger(package).setLevel(logging.INFO)

    if not API_KEYS:
        logger.warning(
            "ADAPTIVE_RAG_API_KEYS is empty: authentication is DISABLED. "
            "Acceptable for local development only."
        )

    # Building the engine loads the tokenizer and models: keep it off the event loop.
    rag = await run_in_threadpool(AdaptiveRAG)
    app.state.rag = rag
    app.state.jobs = SyncJobManager(rag)
    app.state.query_slots = BoundedSemaphore(MAX_CONCURRENT_QUERIES)

    # Index in the background: the server starts answering /health immediately,
    # and /ready turns green once the index is loaded.
    if SYNC_ON_STARTUP:
        app.state.jobs.request()

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Adaptive RAG API",
        version="1.0.0",
        description="Document question answering with verified, cited answers.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s in %.2fs [%s]",
            request.method, request.url.path, response.status_code,
            time.perf_counter() - started, request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error [%s]", request_id)
        # Never leak internals (paths, stack traces, hostnames) to the client.
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    app.include_router(routes_system.router)
    app.include_router(routes_query.router)
    app.include_router(routes_documents.router)
    return app


app = create_app()