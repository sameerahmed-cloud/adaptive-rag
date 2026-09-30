from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from adaptive_rag.engine import AdaptiveRAG

from .dependencies import get_rag_engine
from .schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SyncResponse,
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(rag: AdaptiveRAG = Depends(get_rag_engine)):
    return HealthResponse(
        status="healthy",
        knowledge_base="ready" if rag.vector_index is not None else "not_initialized",
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(
    request: QueryRequest,
    rag: AdaptiveRAG = Depends(get_rag_engine),
):
    try:
        answer = rag.ask(
            request.question,
            rag_mode=request.rag_mode,
            max_retries=request.max_retries,
        )
        return QueryResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {exc}",
        ) from exc


@router.post("/sync", response_model=SyncResponse, tags=["documents"])
def sync(rag: AdaptiveRAG = Depends(get_rag_engine)):
    try:
        rag.sync(force_rebuild=False)
        return SyncResponse(message="Knowledge base synchronized successfully.")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge-base synchronization failed: {exc}",
        ) from exc


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(rag: AdaptiveRAG = Depends(get_rag_engine)):
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                filename=filename,
                document_id=record.get("document_id"),
                size=record.get("size"),
                extension=record.get("extension"),
            )
            for filename, record in sorted(rag.manifest.items())
        ]
    )


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    rag: AdaptiveRAG = Depends(get_rag_engine),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    # Strip client-supplied directories. The first API version only allows
    # uploads directly into the RAG data directory.
    filename = Path(file.filename).name
    destination = rag.data_dir / filename

    if destination.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A document named '{filename}' already exists.",
        )

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        # Reuse the exact synchronization logic already tested in Phase 1–7.
        rag.sync(force_rebuild=False)

        relative_path = str(destination.relative_to(rag.data_dir))
        record = rag.manifest.get(relative_path)
        if record is None:
            raise RuntimeError("Uploaded file was not added to the manifest.")

        return DocumentResponse(
            filename=relative_path,
            document_id=record.get("document_id"),
            size=record.get("size"),
            extension=record.get("extension"),
        )

    except Exception as exc:
        # If ingestion fails before the existing document was indexed, remove
        # the uploaded file so a failed request does not leave an orphan file.
        if destination.exists() and str(destination.relative_to(rag.data_dir)) not in rag.manifest:
            destination.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    tags=["documents"],
)

def delete_document(
    document_id: str,
    rag: AdaptiveRAG = Depends(get_rag_engine),
):
    # 1. Map the document_id to the file path using the current manifest
    matching_path = next(
        (relative_path for relative_path, record in rag.manifest.items() 
         if record.get("document_id") == document_id),
        None,
    )

    if matching_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    try:
        # 2. Remove the physical file FIRST (just like you do when you do it manually!)
        file_path = rag.data_dir / matching_path
        file_path.unlink(missing_ok=True)

        # 3. FORCE the RAG instance to dump its active in-memory LlamaIndex caches.
        # This simulates closing the terminal and opening it fresh.
        if hasattr(rag, "clear_memory_cache"):
            rag.clear_memory_cache()
        else:
            # If you don't have a clear function, reset the storage contexts manually
            rag.vector_index = None
            rag.summary_index = None
            # Force your dependency or initialization logic to reload them from disk
            rag.load_indexes() 

        # 4. Trigger sync. Now it safely calculates (old_paths - current_paths) 
        # exactly like your terminal script does!
        rag.sync(force_rebuild=False)

        return DeleteResponse(message=f"Document '{matching_path}' deleted successfully.")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion failed: {exc}",
        ) from exc
