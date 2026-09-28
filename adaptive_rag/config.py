from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing from .env file.")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY missing from .env file.")

DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
MANIFEST_FILE = STORAGE_DIR / "document_manifest.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_PARSE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".doc"}

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

JSON_EXTENSION = {".json"}
STRUCTURED_EXTENSIONS = {".csv", ".tsv"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}
HTML_EXTENSIONS = {".html", ".htm"}
