import logging
import warnings

logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

from adaptive_rag.cli import run


if __name__ == "__main__":
    run()