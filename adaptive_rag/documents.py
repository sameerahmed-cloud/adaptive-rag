import csv
import hashlib
import os
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from json_repair import repair_json
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.readers.file import HTMLTagReader
from llama_cloud import LlamaCloud

from .config import (
    CODE_LANGUAGES, HTML_EXTENSIONS, LLAMA_PARSE_EXTENSIONS, LLAMA_CLOUD_API_KEY, PARSE_CACHE_DIR, LLAMA_PARSE_TIER,
    MANIFEST_FILE, MARKDOWN_EXTENSIONS, SPREADSHEET_EXTENSIONS, TEXT_EXTENSIONS, STORAGE_DIR, PROFILE_USE_LLM,
)
from .observability import API_TRACKER



@dataclass
class DocumentProfile:

    path: Path
    extension: str
    file_size_bytes: int
    document_type: str = "unknown"
    word_count: int = 0
    line_count: int = 0
    has_headers: bool = False
    has_tables: bool = False
    has_code: bool = False
    structure_depth: int = 0
    language: Optional[str] = None


def calculate_file_hash(
    path: Path,
) -> str:
    """
    Hash the actual file contents.

    Changes when the contents of the file change.
    """

    sha256 = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_document_id(
    relative_path: str,
) -> str:
    """
    Generate a stable document ID from the file path.

    IMPORTANT:

    This does NOT change when the file contents change.
    """

    normalized = relative_path.replace(
        "\\","/",
    ).lower()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_manifest() -> Dict:

    if not MANIFEST_FILE.exists():
        return {}

    try:

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError("Document manifest must contain a JSON object.")

        valid_records = {}
        for relative_path, record in manifest.items():
            if not isinstance(relative_path, str) or not isinstance(record, dict):
                raise ValueError("Document manifest contains an invalid record.")
            if not record.get("document_id") or not record.get("hash"):
                raise ValueError(
                    f"Document manifest record '{relative_path}' is missing document_id/hash."
                )

            canonical_path = relative_path.replace("\\", "/")
            if canonical_path in valid_records:
                raise ValueError(
                    f"Manifest path collision after normalization: '{canonical_path}'."
                )
            valid_records[canonical_path] = record

        return valid_records

    except Exception as error:
        print(f"--> [WARNING] Ignoring invalid document manifest: {error}")
        return {}


def save_manifest(manifest: Dict) -> None:
    
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_file, MANIFEST_FILE)

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def classify_file(path: Path,) -> str:

    extension = path.suffix.lower()

    if extension in LLAMA_PARSE_EXTENSIONS:
        return "document"

    if extension in MARKDOWN_EXTENSIONS:
        return "markdown"

    if extension in CODE_LANGUAGES:
        return "code"

    if extension == ".json":
        return "json"

    if extension in {".csv", ".tsv"}:
        return "tabular"

    if extension in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"

    if extension in HTML_EXTENSIONS:
        return "html"

    if extension in TEXT_EXTENSIONS:
        return "text"

    return "unknown"


def run_local_metrics(text: str) -> Tuple[int, int]:
    
    words = re.findall(r"\b\w+\b", text)
    lines = text.splitlines()
    return len(words), len(lines)

_PROSE_STOPWORDS = r"(?!the\b|a\b|an\b|my\b|our\b|this\b|to\b|of\b|in\b|from\b|with\b)"

_CODE_REGEX = re.compile(
    # A. Multi-Language Structural Signals (Symbols uncommon in plain prose)
    r"^\s*#!/bin/|"                           # Shell scripts (Shebang)
    r"</?([a-zA-Z0-9]+)(?:\s+[^>]*)?>|"       # HTML/XML tags
    r"\b[a-z_][a-zA-Z0-9_]*\s*=>|"             # Arrow functions (Enforces lowercase variable start)
    r":=|->|\|>|"                             # Explicit assignments/pipes (Go, Rust, Elixir)
    r"if\s*\(.*?\)\s*\{|for\s*\(.*?\)\s*\{|"  # C-style control blocks (JS, Java, C++)
    r"^[ \t]*\}\s*$|"                         # Isolated closing brackets standing alone
    
    # B. Strict Keyword Anchors (The Prose Guard)
    # Rules: Must be lowercase, at start of a line (^\s*), and not followed by a stopword.
    r"^\s*(?:def|class|function|var|let|const|return)\b\s+" + _PROSE_STOPWORDS + r"[a-zA-Z_]|"
    r"^\s*import\b\s+" + _PROSE_STOPWORDS + r"[a-zA-Z0-9_{]|"
    r"^\s*from\b\s+[a-zA-Z0-9_.]+\s+import\b", # Python style imports
    
    flags=re.MULTILINE
)

# SQL-specific structures (Kept separate to allow case-insensitive handling for standard query layouts)
_SQL_REGEX = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b\s+(?:FROM|INTO|TABLE|DATABASE|\*)\b",
    flags=re.IGNORECASE | re.MULTILINE
)

# Structural Data Tables
_TABLE_REGEX = re.compile(
    r"\|.*\||"                        # Markdown tables
    r"^[^\t\n]+(?:\t[^\t\n]+){2,}|"   # Tab-Separated Values (TSV)
    r"^[^,\n\s]+(?:,[^,\n\s]+){3,}",  # Comma-Separated Values (CSV - enforces no structural prose spaces)
    flags=re.MULTILINE
)

# Document Headers
_HEADER_REGEX = re.compile(
    r"^#{1,6}\s+|"                    # Markdown headers
    r"^\d+(?:\.\d+)*\s+[A-Z]",        # Numbered sections starting with an uppercase letter
    flags=re.MULTILINE
)

def run_heuristic_filter(text: str) -> dict[str, bool]:
    """Runs high-performance local regex heuristics on document text chunks.
    
    Acts as a linear-time safety gatekeeper to detect code, data tables, 
    and structural headers before triggering heavy downstream RAG parsers 
    or LLM profiling.
    """
    if not text or not text.strip():
        return {
            "needs_code_check": False,
            "needs_table_check": False,
            "needs_header_check": False,
        }

    has_code_signals = bool(_CODE_REGEX.search(text)) or bool(_SQL_REGEX.search(text))
    has_table_signals = bool(_TABLE_REGEX.search(text))
    has_header_signals = bool(_HEADER_REGEX.search(text))

    return {
        "needs_code_check": has_code_signals,
        "needs_table_check": has_table_signals,
        "needs_header_check": has_header_signals,
    }


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Safely converts common JSON-ish string or numeric values into primitive Booleans.
    
    Excludes Python floats from accidental matching to protect upstream metadata tracking.
    """
    if isinstance(value, bool):
        return value
        
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
        
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
            
    return default


def evaluate_content_structure_llm(text: str, filters: dict[str, bool]) -> dict[str, bool]:
    """Uses an LLM to validate complex metadata layout configurations dynamically."""
    structure = {"has_headers": False, "has_tables": False, "has_code": False}

    mapping = {
        "needs_header_check": ("has_headers", "1. Structural headings, section titles, or chapter lines."),
        "needs_table_check": ("has_tables", "2. Data grids, tables, CSV formatting, or spreadsheet layouts."),
        "needs_code_check": ("has_code", "3. Functional blocks of source code, configurations, or scripting syntax.")
    }

    active_instructions, keys_to_evaluate = [], []
    for filter_key, (struct_key, instruction) in mapping.items():
        if filters.get(filter_key, False):
            active_instructions.append(instruction)
            keys_to_evaluate.append(struct_key)

    if not active_instructions or not text.strip():
        return structure

    snippet = f"{text[:2000]}\n\n[...]\n\n{text[-2000:]}" if len(text) > 4000 else text
    schema_template = {k: "true/false" for k in keys_to_evaluate}
    
    instructions_text = "\n".join(active_instructions)
    
    prompt = f"""
    Analyze the following text snippet from a document. Determine if it contains any of these specific elements:
    {instructions_text}

    Respond STRICTLY in valid raw JSON format matching this specific schema constraint:
    {json.dumps(schema_template, indent=2)}

    Text Snippet:
    \"\"\"{snippet}\"\"\"
    """

    try:
        llm_before = API_TRACKER.llm_snapshot()
        response = Settings.llm.complete(prompt)
        API_TRACKER.record_llm_operation("document_profiling", llm_before)
        
        clean_response = repair_json(response.text.strip())
        llm_analysis = json.loads(clean_response)
        
        for key in keys_to_evaluate:
            structure[key] = _coerce_bool(llm_analysis.get(key), default=False)
    except Exception as error:
        print(f"--> [PROFILE WARNING] Structure analysis failed: {error}")

    return structure

# Fenced code blocks are removed before scanning so a "# comment" inside a
# ```python block is not mistaken for a markdown heading.
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
 
# "# Title", "## Section": 1-6 hashes at the START of a line, then a space,
# then real text. "#hashtag", "C# is..." and "a # b" do not match.
_MD_HEADER = re.compile(r"^(#{1,6})[ \t]+\S", re.MULTILINE)
 
# The row under a markdown table header: |---|---| or | :--- | ---: |
# A line of only dashes/colons/spaces with at least one pipe separator.
# This is what proves a table (a prose line with two pipes does not match).
_MD_TABLE_SEPARATOR = re.compile(
    r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$",
    re.MULTILINE,
)
 
# Outline numbering like "1.4.2 Scope". Used for structure_depth only.
_NUMERIC_HEADER_REGEX = re.compile(r"^(\d+(?:\.\d+)*)[ \t]+[A-Z]", re.MULTILINE)

_LLM_PROFILE_CACHE: Dict[str, Dict[str, bool]] = {}
 
 
def _llm_structure_cached(text: str, filters: dict) -> Dict[str, bool]:
    """Ask the LLM at most once per (text, question) pair."""
    key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    key += "|" + ",".join(sorted(k for k, v in filters.items() if v))
 
    if key not in _LLM_PROFILE_CACHE:
        _LLM_PROFILE_CACHE[key] = evaluate_content_structure_llm(text, filters)
 
    return _LLM_PROFILE_CACHE[key]
 
 
def profile_document(path: Path, extracted_text: Optional[str] = None) -> DocumentProfile:
    """Profile a document deterministically; optionally consult an LLM.
 
    Order of evidence (cheapest and most reliable first):
      Tier 1  file extension / classified type
      Tier 2  exact markdown syntax in the extracted text
      Tier 3  (optional, PROFILE_USE_LLM) LLM, only for what Tier 1-2 left open
    """
    extension = path.suffix.lower()
    document_type = classify_file(path)
 
    profile = DocumentProfile(
        path=path,
        extension=extension,
        file_size_bytes=path.stat().st_size,
        document_type=document_type,
    )
 
    # ---- Tier 1: the extension already answers some questions ----
    if extension in CODE_LANGUAGES:
        profile.language = CODE_LANGUAGES[extension]
        profile.has_code = True
 
    if document_type in {"tabular", "spreadsheet"}:
        profile.has_tables = True
 
    if not extracted_text:
        return profile
 
    profile.word_count, profile.line_count = run_local_metrics(extracted_text)
 
    # Source code: '#' lines are comments, not headings. Nothing more to learn.
    if document_type == "code":
        return profile
 
    # ---- Tier 2: exact markdown syntax (LlamaParse returns markdown) ----
    scan = _FENCED_CODE.sub("", extracted_text)
 
    md_levels = [len(m) for m in _MD_HEADER.findall(scan)]
    profile.has_headers = bool(md_levels)
 
    if not profile.has_tables:
        profile.has_tables = (
            bool(_MD_TABLE_SEPARATOR.search(scan)) or "<table" in scan.lower()
        )
 
    if not profile.has_code:
        profile.has_code = "```" in extracted_text
 
    numeric_levels = [
        len(m.split(".")) for m in _NUMERIC_HEADER_REGEX.findall(scan)
    ]
    profile.structure_depth = max(md_levels + numeric_levels, default=0)
 
    # ---- Tier 3: optional LLM, only for unresolved questions ----
    if PROFILE_USE_LLM and document_type in {"text", "unknown"}:
        filters = run_heuristic_filter(extracted_text)
        filters["needs_header_check"] &= not profile.has_headers
        filters["needs_table_check"] &= not profile.has_tables
        filters["needs_code_check"] &= not profile.has_code
 
        if any(filters.values()):
            insights = _llm_structure_cached(extracted_text, filters)
            profile.has_headers = profile.has_headers or insights["has_headers"]
            profile.has_tables = profile.has_tables or insights["has_tables"]
            profile.has_code = profile.has_code or insights["has_code"]
            if profile.has_headers and profile.structure_depth == 0:
                profile.structure_depth = 1
 
    return profile


def parse_with_llama_cloud(path: Path) -> List[Document]:
    """Parse one file with the current LlamaParse API (v2). One Document per page."""
    if not LLAMA_CLOUD_API_KEY:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is required for complex document parsing. "
            "Add it to the environment or .env file."
        )

    # The client reads LLAMA_CLOUD_API_KEY from the environment (config.py loads
    # .env into it). One client per call, so parallel ingestion threads do not
    # share a single object.
    client = LlamaCloud()

    uploaded = client.files.create(file=str(path), purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded.id,
        tier=LLAMA_PARSE_TIER,
        version="latest",
        expand=["markdown"],
    )

    pages = result.markdown.pages
    failed = [page.page_number for page in pages if not page.success]
    if failed:
        print(f"--> [WARNING] {path.name}: {len(failed)} page(s) failed to parse: {failed[:10]}")

    return [
        Document(text=page.markdown, metadata={"page_label": str(page.page_number)})
        for page in pages
        if page.success and page.markdown and page.markdown.strip()
    ]


def attach_document_identity(
    document: Document,
    path: Path,
    document_id: str,
    file_hash: str,
    part_index: int = 0,
):

    document.doc_id = document_id

    document.metadata.update(
        {
            "document_id": document_id,
            "file_hash": file_hash,
            "file_name": path.name,
            "file_path": str(path),
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "part_index": part_index
        }
    )

PARSE_CACHE_VERSION = f"cloud-v2-{LLAMA_PARSE_TIER}"

def prune_parse_cache(live_hashes: set, max_age_days: int = 30) -> int:
    """Delete cache entries for files no longer indexed (and old-version entries)."""
    if not PARSE_CACHE_DIR.exists():
        return 0
    removed, cutoff = 0, time.time() - max_age_days * 86400
    for path in PARSE_CACHE_DIR.glob("*.json"):
        stale_version = not path.name.endswith(f".{PARSE_CACHE_VERSION}.json")
        orphaned = path.name.split(".")[0] not in live_hashes and path.stat().st_mtime < cutoff
        if stale_version or orphaned:
            path.unlink()
            removed += 1
    return removed

def _parse_cache_path(file_hash: str) -> Path:
    return PARSE_CACHE_DIR / f"{file_hash}.{PARSE_CACHE_VERSION}.json"
 
 
def load_cached_parse(file_hash: str) -> Optional[List[Document]]:
    """Return previously parsed documents for this exact file content, or None."""
    try:
        data = json.loads(_parse_cache_path(file_hash).read_text(encoding="utf-8"))
        documents = [
            Document(text=item["text"], metadata=item.get("metadata", {}))
            for item in data
            if item.get("text", "").strip()
        ]
        return documents or None
    except Exception:
        return None  # missing or corrupt cache entry: just parse again
 
 
def save_cached_parse(file_hash: str, documents: List[Document]) -> None:
    """Store parser output. Failures are logged, never raised."""
    if not documents:
        return
    path = _parse_cache_path(file_hash)
    temp = path.with_suffix(".tmp")
    try:
        PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = [
            {"text": doc.text, "metadata": doc.metadata} for doc in documents
        ]
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temp, path)  # atomic: a reader never sees half a file
    except Exception as error:
        print(f"--> [WARNING] Could not cache parse result: {error}")
        if temp.exists():
            temp.unlink()
 

def load_single_file(
    path: Path,
    document_id: str,
    file_hash: str,
) -> List[Document]:
    """Loads text vectors, converts binary records, and profiles chunk structural geometry.
    
    Acts as the entry interface for data ingestion, ensuring that every loaded document block
    emerges fully enriched with structural, programmatic, and layout analytics.
    """
    extension = path.suffix.lower()
    document_type = classify_file(path)
    parsed_documents: List[Document] = []

    if extension in LLAMA_PARSE_EXTENSIONS:
        parsed_documents = load_cached_parse(file_hash) or []
        if parsed_documents:
            print(f"--> [PARSE CACHE] {path.name}: reused previous parse")
        else:
            parse_start = time.perf_counter()
            parsed_documents = parse_with_llama_cloud(path)
            API_TRACKER.record_llama_parse(time.perf_counter() - parse_start)
            save_cached_parse(file_hash, parsed_documents)

    elif extension in MARKDOWN_EXTENSIONS or extension in CODE_LANGUAGES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        parsed_documents = [Document(text=text)]

    elif extension == ".json":
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            text = path.read_text(encoding="utf-8", errors="ignore")
        parsed_documents = [Document(text=text)]

    elif extension in {".csv", ".tsv"}:
        delimiter = "\t" if extension == ".tsv" else ","
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                reader = csv.reader(file, delimiter=delimiter)
                for row in reader:
                    rows.append(" | ".join(str(value) for value in row))
        except Exception:
            rows = [path.read_text(encoding="utf-8", errors="ignore")]
        parsed_documents = [Document(text="\n".join(rows))]

    elif extension in SPREADSHEET_EXTENSIONS:
        return load_excel_file(path, document_id, file_hash)

    elif extension in HTML_EXTENSIONS:
        reader = HTMLTagReader()
        parsed_documents = reader.load_data(file=path)

    elif document_type == "unknown":
        try:
            fallback_docs = SimpleDirectoryReader(input_files=[str(path)], filename_as_id=False).load_data()
            for doc in fallback_docs:
                if getattr(doc, "text", "").strip():
                    parsed_documents.append(doc)
        except Exception as error:
            print(f"--> [SKIP] {path.name}: No compatible binary reader found ({error})")
            return []

    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                parsed_documents = [Document(text=text)]
        except Exception as error:
            print(f"--> [SKIP] {path.name}: File reading error exception ({error})")
            return []

    final_validated_documents = []

    part_index = 0
    for document in parsed_documents:
        if not document.text or not document.text.strip():
            continue

        attach_document_identity(document, path, document_id, file_hash, part_index)
        part_index += 1

        document.metadata["document_type"] = document_type
        if extension in CODE_LANGUAGES:
            document.metadata["language"] = CODE_LANGUAGES[extension]

        final_validated_documents.append(document)

    return final_validated_documents

def load_excel_file(
    path: Path,
    document_id: str,
    file_hash: str,
) -> List[Document]:
    """Loads an Excel workbook sheet by sheet and serializes grids into clean Markdown rows."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError("Install openpyxl to process Excel files.")

    documents = []

    # Use read_only=True and data_only=True for maximum parsing velocity and minimal RAM usage
    workbook = load_workbook(
        filename=path,
        read_only=True,
        data_only=True,
    )

    try:
        for worksheet in workbook.worksheets:
            rows = []

            for row in worksheet.iter_rows(values_only=True):
                # Optimization: Strip out None values and right-strip trailing empty entries
                values = [str(cell) for cell in row if cell is not None]

                # Only include the line if it contains meaningful data
                if values and any(v.strip() for v in values):
                    rows.append(" | ".join(values))

            text = "\n".join(rows)
            if not text.strip():
                continue

            document = Document(text=text)
            attach_document_identity(document, path, document_id, file_hash, len(documents))

            # Metadata hints pass a Tier 1 shortcut context directly to your downstream chunker
            document.metadata.update({
                "document_type": "spreadsheet",
                "sheet_name": worksheet.title,
            })

            documents.append(document)

    finally:
        workbook.close()

    return documents


def discover_files(
    data_dir: Path,
) -> List[Path]:
    """Return user documents while excluding hidden files and RAG storage.

    """
    data_dir = data_dir.resolve()
    storage_dir = STORAGE_DIR.resolve()
    files = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(data_dir)

        if any(part.startswith(".") for part in relative.parts):
            continue

        resolved = path.resolve()
        if resolved == storage_dir or storage_dir in resolved.parents:
            continue

        files.append(path)

    return sorted(files)
