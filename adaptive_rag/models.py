from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .observability import API_TRACKER


class TimedHuggingFaceEmbedding(HuggingFaceEmbedding):
    """Time the actual local HuggingFace embedding operations."""

    def _get_text_embeddings(self, texts):
        import time
        start = time.perf_counter()
        try:
            return super()._get_text_embeddings(texts)
        finally:
            API_TRACKER.record_embedding_time(time.perf_counter() - start)

    async def _aget_text_embeddings(self, texts):
        import time
        start = time.perf_counter()
        try:
            return await super()._aget_text_embeddings(texts)
        finally:
            API_TRACKER.record_embedding_time(time.perf_counter() - start)

    def _get_query_embedding(self, query):
        import time
        start = time.perf_counter()
        try:
            return super()._get_query_embedding(query)
        finally:
            API_TRACKER.record_query_embedding_time(time.perf_counter() - start)

    async def _aget_query_embedding(self, query):
        import time
        start = time.perf_counter()
        try:
            return await super()._aget_query_embedding(query)
        finally:
            API_TRACKER.record_query_embedding_time(time.perf_counter() - start)


_configured = False


def configure_models():
    """Configure the project's LLM, embedding model, and callback instrumentation once."""
    global _configured
    if _configured:
        return

    API_TRACKER.attach()
    Settings.llm = GoogleGenAI(model="models/gemini-3.1-flash-lite")
    Settings.embed_model = TimedHuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    _configured = True
