"""Configuration for the Adaptive RAG application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Google currently accepts GEMINI_API_KEY and GOOGLE_API_KEY.  We keep the
# project's existing GEMINI_API_KEY name while supporting Google's alternate
# environment variable as a compatibility fallback.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

LLAMA_PARSE_TIER = "cost_effective"

# Runtime configuration is environment-driven so the same code can run
# locally, in Docker, or in a hosted environment without editing source code.
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "pipeline_collection"
GEMINI_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_EMBEDDING_TOKENS = 512
EMBEDDING_SPECIAL_TOKENS = 2
EMBEDDING_METADATA_RESERVE = 48
FALLBACK_TOKEN_MARGIN = 0.75

SHORT_TEXT_MAX_WORDS = 350    
MIN_CHUNK_TOKENS = 64             
CHUNK_OVERLAP_RATIO = 0.10
MIN_CHUNK_OVERLAP = 30
CHARS_PER_TOKEN_CODE = 3

CHUNK_SIZE_TIERS = [(5000, 400), (20000, 375), (100000, 350)]
CHUNK_SIZE_DEFAULT = 300

CODE_LINE_TIERS = [(100, 80), (500, 60), (2000, 50), (5000, 40)]
CODE_LINE_DEFAULT = 30

EMBED_METADATA_KEYS = {"file_name", "sheet_name", "header_path", "page_label"}
LLM_METADATA_KEYS = {"file_name", "sheet_name", "header_path", "page_label"}

MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

DATA_DIR = Path( "./data")
STORAGE_DIR = Path("./storage")
MANIFEST_FILE = STORAGE_DIR / "document_manifest.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# LlamaParse is the managed parser for complex/unstructured files.  Image
# formats are included because the current LlamaParse platform is designed for
# scans and document images as well as office documents.
LLAMA_PARSE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".pptm",
    ".png",
    ".jpg",
    ".jpeg",
}

MARKDOWN_EXTENSIONS = {".md", ".markdown"}

CODE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
    ".h": "c", ".hpp": "cpp", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin", ".cs": "csharp",
    ".scala": "scala", ".sql": "sql", ".sh": "bash", ".bash": "bash",
}

TEXT_EXTENSIONS = {
    ".txt", ".text", ".log", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".yaml", ".yml", ".toml",
}

SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}
HTML_EXTENSIONS = {".html", ".htm"}

PROFILE_USE_LLM = False

RETRIEVAL_TOP_K = 5          # chunks sent to the LLM
CANDIDATE_TOP_K = 12         # chunks fetched per retriever before reranking
MAX_EXPANSION_NODES = 4      # max neighbour chunks added around boundary hits
SUMMARY_MAX_NODES = 12       # hard cap on chunks used for "summary" questions

MIN_RETRIEVAL_SCORE = None

MAX_QUESTION_CHARS = 2000    # longer questions are rejected before any model call
REQUIRE_CITATIONS = True     # answers must cite supporting chunks as [Context N]
LLM_RETRY_ATTEMPTS = 2       # transient LLM failures are retried with backoff

INGEST_WORKERS = 1           # files parsed/chunked in parallel (1 = sequential)
MAX_INGEST_ATTEMPTS = 3      # a failing file is retried this many times, then
                             # skipped until its content changes
PARSE_CACHE_DIR = Path("./cache") / "parsed"


# Comma-separated keys, e.g.  ADAPTIVE_RAG_API_KEYS=key-for-web-app,key-for-admin
# Leave EMPTY only on your own machine: with no keys, authentication is off.
API_KEYS = [
    key.strip()
    for key in os.getenv("ADAPTIVE_RAG_API_KEYS", "").split(",")
    if key.strip()
]
 
# Origins allowed to call the API from a browser (your web app's address).
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ADAPTIVE_RAG_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
 
API_HOST = os.getenv("ADAPTIVE_RAG_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("ADAPTIVE_RAG_API_PORT", "8000"))
 
SYNC_ON_STARTUP = True             # index the data folder when the server starts
ASK_RATE_LIMIT_PER_MINUTE = 3     # per client; 0 disables
MAX_CONCURRENT_QUERIES = 4         # questions processed at once
QUERY_QUEUE_TIMEOUT_SECONDS = 20   # how long a question waits for a free slot