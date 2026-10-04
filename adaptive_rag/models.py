import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from .config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_EMBEDDING_TOKENS,
)
from .observability import API_TRACKER

# How many timed calls are currently open in this thread / async task.
# ContextVar (not threading.local) so concurrent asyncio tasks on one thread
# do not see each other's depth.
_timing_depth: ContextVar[int] = ContextVar("embedding_timing_depth", default=0)


@contextmanager
def _timed(record):
    """Time a block and report it, but only for the OUTERMOST timed call.

    If one timed method calls another (an async wrapper delegating to the sync
    one, or the reverse), recording both would double-count the same work and
    inflate both the duration and the batch counter.
    """
    depth = _timing_depth.get()
    token = _timing_depth.set(depth + 1)
    start = time.perf_counter()
    try:
        yield
    finally:
        _timing_depth.reset(token)
        if depth == 0:
            record(time.perf_counter() - start)


class TimedHuggingFaceEmbedding(HuggingFaceEmbedding):
    """Time the actual local HuggingFace embedding operations."""

    def _get_text_embeddings(self, texts):
        with _timed(API_TRACKER.record_embedding_time):
            return super()._get_text_embeddings(texts)

    async def _aget_text_embeddings(self, texts):
        with _timed(API_TRACKER.record_embedding_time):
            return await super()._aget_text_embeddings(texts)

    def _get_query_embedding(self, query):
        with _timed(API_TRACKER.record_query_embedding_time):
            return super()._get_query_embedding(query)

    async def _aget_query_embedding(self, query):
        with _timed(API_TRACKER.record_query_embedding_time):
            return await super()._aget_query_embedding(query)


_configured = False
_configure_lock = threading.Lock()


def configure_models():
    """Configure the project's LLM, embedding model, and callback instrumentation once."""
    global _configured

    # Two threads starting up together (for example two first requests) must
    # not both run the setup.
    with _configure_lock:
        if _configured:
            return

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. "
                "Add it to the environment or .env file before starting the RAG engine."
            )

        # Must happen BEFORE the models are assigned: Settings hands its
        # current callback manager to each model as it is set, which is how
        # token usage gets recorded.
        API_TRACKER.attach()

        Settings.llm = GoogleGenAI(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
        )

        Settings.embed_model = TimedHuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            # Same number the chunker budgets against, so the embedder and the
            # chunker cannot silently disagree about the window.
            max_length=MAX_EMBEDDING_TOKENS,
        )

        _configured = True