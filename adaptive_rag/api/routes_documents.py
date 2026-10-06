import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile

from ..config import (
    CODE_LANGUAGES, HTML_EXTENSIONS, LLAMA_PARSE_EXTENSIONS, MARKDOWN_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES, SPREADSHEET_EXTENSIONS, TEXT_EXTENSIONS,
)
from ..engine import AdaptiveRAG
from .deps import get_jobs, get_rag
from .jobs import SyncJobManager
from .schemas import DocumentOut, UploadResponse
from .security import require_api_key, safe_filename

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)

# Only file types the loader knows how to read.
ALLOWED_EXTENSIONS = (
    set(LLAMA_PARSE_EXTENSIONS) | set(MARKDOWN_EXTENSIONS) | set(CODE_LANGUAGES)
    | set(TEXT_EXTENSIONS) | set(SPREADSHEET_EXTENSIONS) | set(HTML_EXTENSIONS)
    | {".json", ".csv", ".tsv"}
)


@router.get("", response_model=List[DocumentOut])
def list_documents(rag: AdaptiveRAG = Depends(get_rag)) -> List[DocumentOut]:
    """Every known document and whether it is indexed. Never returns server paths."""
    documents = []
    for path, record in sorted(rag.manifest_snapshot().items()):
        failure = record.get("failure") or {}
        documents.append(
            DocumentOut(
                path=path,
                status="failed" if record.get("status") == "failed" else "indexed",
                extension=record.get("extension"),
                size_bytes=record.get("size"),
                indexed_nodes=int(record.get("indexed_nodes", 0) or 0),
                error=failure.get("error"),
                attempts=failure.get("attempts"),
            )
        )
    return documents


def _save_stream(upload: UploadFile, target: Path, data_dir: Path) -> int:
    """Write the upload to a hidden temp file, then move it into place.

    The sync ignores hidden files, so it can never read a half-written upload,
    and the final os.replace is atomic.
    """
    temp = data_dir / f".upload-{uuid.uuid4().hex}.tmp"
    size = 0
    try:
        with open(temp, "wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                out.write(chunk)

        if size == 0:
            raise HTTPException(status_code=422, detail="The uploaded file is empty.")

        try:
            os.replace(temp, target)
        except PermissionError:
            raise HTTPException(
                status_code=409,
                detail="That file is being processed right now. Try again in a moment.",
            )
        return size
    finally:
        temp.unlink(missing_ok=True)


@router.post("", response_model=UploadResponse, status_code=202)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    overwrite: bool = Query(True, description="Replace an existing file with the same name."),
    sync: bool = Query(True, description="Start indexing right after the upload."),
    rag: AdaptiveRAG = Depends(get_rag),
    jobs: SyncJobManager = Depends(get_jobs),
) -> UploadResponse:
    """Upload a document. Indexing happens in the background (HTTP 202):
    poll GET /sync or GET /documents to see when it is searchable."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_SIZE_BYTES + 1_000_000:
        raise HTTPException(status_code=413, detail="File is too large.")

    try:
        name = safe_filename(file.filename or "")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension or 'none'}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    target = rag.data_dir / name
    existed = target.exists()
    if existed and not overwrite:
        raise HTTPException(status_code=409, detail=f"'{name}' already exists.")

    size = _save_stream(file, target, rag.data_dir)

    job = jobs.request() if sync else None
    return UploadResponse(path=name, size_bytes=size, replaced=existed, sync=job)


@router.delete("/{relative_path:path}", status_code=204)
def delete_document(
    relative_path: str,
    rag: AdaptiveRAG = Depends(get_rag),
    jobs: SyncJobManager = Depends(get_jobs),
) -> Response:
    """Remove a document from the index and delete its file."""
    if jobs.is_running():
        # remove_document waits for the running sync; tell the client instead
        # of leaving the request hanging for minutes.
        raise HTTPException(status_code=409, detail="A sync is in progress. Try again when it finishes.")

    try:
        removed = rag.remove_document(relative_path, delete_file=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if not removed:
        raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=204)