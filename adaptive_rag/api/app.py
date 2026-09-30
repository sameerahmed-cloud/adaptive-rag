from contextlib import asynccontextmanager

from fastapi import FastAPI

from adaptive_rag.engine import AdaptiveRAG

from .routes import router

import logging
import warnings

logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize one shared RAG engine for this API process."""
    rag = AdaptiveRAG(data_dir="./data")
    rag.sync(force_rebuild=False)
    app.state.rag = rag

    yield

    # There is currently no explicit close() method on AdaptiveRAG.
    # If long-lived resources gain explicit cleanup later, close them here.
    app.state.rag = None


app = FastAPI(
    title="Adaptive RAG API",
    version="1.0.0",
    description="HTTP API for the Adaptive RAG knowledge base.",
    lifespan=lifespan,
)

app.include_router(router)
