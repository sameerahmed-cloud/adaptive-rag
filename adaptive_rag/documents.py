import csv
import hashlib
import os
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from json_repair import repair_json
from llama_index.core import Document, Settings
from llama_index.readers.file import HTMLTagReader
from llama_parse import LlamaParse

from .config import (
    CODE_LANGUAGES, DATA_DIR, HTML_EXTENSIONS, LLAMA_PARSE_EXTENSIONS,
    MANIFEST_FILE, MARKDOWN_EXTENSIONS, SPREADSHEET_EXTENSIONS, STORAGE_DIR,
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
        "\\",
        "/",
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

            return json.load(file)

    except Exception:

        return {}


def save_manifest(
    manifest: Dict,
):

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = MANIFEST_FILE.with_suffix(
        ".tmp"
    )

    try:
        
        with open(
            temp_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temp_file,
            MANIFEST_FILE,
        )

    except Exception:

        if temp_file.exists():
            temp_file.unlink()

        raise


def classify_file(
    path: Path,
) -> str:

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


def run_heuristic_filter(text: str) -> dict:

    has_code_signals = bool(re.search(
        r"\b(def|class|import|function|SELECT|FROM)\b|=>|if\s*\(", 
        text, 
        flags=re.IGNORECASE
    ))
    
    has_table_signals = bool(re.search(
        r"\||(?m)^[^,\n]*(?:,[^,\n]*){3,}|(?m)^[^\t\n]*(?:\t[^\t\n]*){3,}", 
        text
    ))
    
    has_header_signals = bool(re.search(
        r"(?m)^#{1,6}\s+|(?m)^\d+(\.\d+)*\s+[A-Z]", 
        text
    ))

    return {
        "needs_code_check": has_code_signals,
        "needs_table_check": has_table_signals,
        "needs_header_check": has_header_signals
    }


def evaluate_content_structure_llm(text: str, filters: dict) -> dict:
   
    structure = {
        "has_headers": False,
        "has_tables": False,
        "has_code": False
    }

    checks_to_perform = []
    if filters["needs_header_check"]:
        checks_to_perform.append("1. Structural headings, section titles, or chapter lines.")
    if filters["needs_table_check"]:
        checks_to_perform.append("2. Data grids, tables, CSV formatting, or spreadsheet layouts.")
    if filters["needs_code_check"]:
        checks_to_perform.append("3. Functional blocks of source code, configurations, or scripting syntax.")

    if not checks_to_perform or not text.strip():
        return structure

    snippet = text[:2000]
    checks_str = "\n".join(checks_to_perform)
    
    prompt = f"""
    Analyze the following text snippet from a document. Determine if it contains any of these specific elements:
    {checks_str}

    Respond STRICTLY in valid raw JSON format matching this schema without any markdown formatting wrappers:
    {{
        "has_headers": true/false,
        "has_tables": true/false,
        "has_code": true/false
    }}

    Text Snippet:
    \"\"\"{snippet}\"\"\"
    """

    try:
        llm_before = API_TRACKER.llm_snapshot()
        response = Settings.llm.complete(prompt)
        API_TRACKER.record_llm_operation(
            "document_profiling",
            llm_before,
        )
        clean_response = repair_json(response.text.strip())
        
        
        llm_analysis = json.loads(clean_response)
        structure.update(llm_analysis)
    except Exception:
        pass

    return structure


def profile_document(
    path: Path,
    extracted_text: Optional[str] = None,
) -> DocumentProfile:
    
    extension = path.suffix.lower()
    document_type = classify_file(path)

    profile = DocumentProfile(
        path=path,
        extension=extension,
        file_size_bytes=path.stat().st_size,
        document_type=document_type,
    )

    if extension in CODE_LANGUAGES:
        profile.language = CODE_LANGUAGES[extension]
        profile.has_code = True

    if extracted_text:
        profile.word_count, profile.line_count = run_local_metrics(extracted_text)

        filters = run_heuristic_filter(extracted_text)
        
        if filters["needs_code_check"] or filters["needs_table_check"] or filters["needs_header_check"]:
            llm_insights = evaluate_content_structure_llm(extracted_text, filters)
            
            profile.has_headers = llm_insights["has_headers"]
            profile.has_tables = llm_insights["has_tables"]
            if not profile.has_code:
                profile.has_code = llm_insights["has_code"]
        else:
            profile.has_headers = False
            profile.has_tables = False

        header_matches = re.findall(r"(?m)^#{1,6}\s+", extracted_text)
        if header_matches:
            profile.structure_depth = max(len(header.strip()) for header in header_matches)
        elif profile.has_headers:
            profile.structure_depth = 1

    return profile


def create_llama_parser():

    return LlamaParse(
        result_type="markdown",
        verbose=False,
    )


def attach_document_identity(
    document: Document,
    path: Path,
    document_id: str,
    file_hash: str,
):

    document.doc_id = document_id

    document.metadata.update(
        {
            "document_id": document_id,
            "file_hash": file_hash,
            "file_name": path.name,
            "file_path": str(path),
        }
    )


def load_single_file(
    path: Path,
    document_id: str,
    file_hash: str,
) -> List[Document]:

    extension = path.suffix.lower()

    document_type = classify_file(
        path
    )

    if extension in LLAMA_PARSE_EXTENSIONS:

        parser = create_llama_parser()

        parse_start = time.perf_counter()
        parsed_documents = parser.load_data(
            str(path)
        )
        API_TRACKER.record_llama_parse(
            time.perf_counter() - parse_start
        )

        for document in parsed_documents:

            attach_document_identity(
                document,
                path,
                document_id,
                file_hash,
            )

            document.metadata[
                "document_type"
            ] = document_type

        return parsed_documents

    if extension in MARKDOWN_EXTENSIONS:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        document = Document(
            text=text,
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata[
            "document_type"
        ] = document_type

        return [document]

    if extension in CODE_LANGUAGES:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        document = Document(
            text=text,
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata.update(
            {
                "document_type": document_type,
                "language": CODE_LANGUAGES[
                    extension
                ],
            }
        )

        return [document]

    if extension == ".json":

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            text = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

        except Exception:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        document = Document(
            text=text,
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata[
            "document_type"
        ] = document_type

        return [document]

    if extension in {
        ".csv",
        ".tsv",
    }:

        delimiter = (
            "\t"
            if extension == ".tsv"
            else ","
        )

        rows = []

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as file:

                reader = csv.reader(
                    file,
                    delimiter=delimiter,
                )

                for row in reader:

                    rows.append(
                        " | ".join(
                            str(value)
                            for value in row
                        )
                    )

        except Exception:

            rows = [
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            ]

        document = Document(
            text="\n".join(rows)
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata[
            "document_type"
        ] = document_type

        return [document]

    if extension in SPREADSHEET_EXTENSIONS:

        return load_excel_file(
            path,
            document_id,
            file_hash,
        )

    if extension in HTML_EXTENSIONS:
        
        reader = HTMLTagReader()
        parsed_documents = reader.load_data(file=path)
        
        for document in parsed_documents:
            attach_document_identity(
                document,
                path, 
                document_id, 
                file_hash
            )
            document.metadata[
                "document_type"
            ] = document_type

        return parsed_documents

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if not text.strip():
            return []

        document = Document(
            text=text,
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata[
            "document_type"
        ] = document_type

        return [document]

    except Exception as error:

        print(
            f"--> [SKIP] {path.name}: {error}"
        )

        return []


def load_excel_file(
    path: Path,
    document_id: str,
    file_hash: str,
) -> List[Document]:

    try:

        from openpyxl import load_workbook

    except ImportError:

        raise ImportError(
            "Install openpyxl to process Excel files."
        )

    documents = []

    workbook = load_workbook(
        filename=path,
        read_only=True,
        data_only=True,
    )

    for worksheet in workbook.worksheets:

        rows = []

        for row in worksheet.iter_rows(
            values_only=True
        ):

            values = [
                ""
                if value is None
                else str(value)
                for value in row
            ]

            if any(values):

                rows.append(
                    " | ".join(values)
                )

        text = "\n".join(rows)

        if not text.strip():
            continue

        document = Document(
            text=text
        )

        attach_document_identity(
            document,
            path,
            document_id,
            file_hash,
        )

        document.metadata.update(
            {
                "document_type": "spreadsheet",
                "sheet_name": worksheet.title,
            }
        )

        documents.append(
            document
        )

    return documents


def discover_files(
    data_dir: Path,
) -> List[Path]:

    files = []

    for path in data_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if STORAGE_DIR in path.parents:
            continue

        files.append(path)

    return sorted(files)
