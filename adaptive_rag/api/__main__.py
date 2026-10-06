import uvicorn
import logging
import warnings

from ..config import API_HOST, API_PORT


logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    # workers=1 is deliberate: the engine holds the index, locks and caches in
    # this process. Several workers would each load their own copy and run
    # their own syncs against the same Qdrant collection.
    uvicorn.run("adaptive_rag.api.app:app", host=API_HOST, port=API_PORT, workers=1)