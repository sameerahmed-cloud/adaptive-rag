from fastapi import Request

from adaptive_rag.engine import AdaptiveRAG


def get_rag_engine(request: Request) -> AdaptiveRAG:
    """Provide the application-wide RAG engine to an API route."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise RuntimeError("RAG engine is not initialized.")
    return rag
