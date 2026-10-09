"""Documents API: upload, list, and delete indexed documents."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.retrieval.vector_store import delete_document_chunks
from app.schemas import DeleteResponse, DocumentListResponse, UploadResponse
from app.services.document_registry import (
    get_document,
    list_documents,
    remove_document,
)
from app.services.ingestion_service import ingest_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
)
async def upload_document(file: Annotated[UploadFile, File()]) -> UploadResponse:
    """Accept a PDF, TXT, or Markdown file, parse it, and index it in ChromaDB."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    content = await file.read()
    try:
        result = ingest_file(file.filename, content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return result


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
def list_all_documents() -> DocumentListResponse:
    """Return all documents currently registered in the service."""
    docs = list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its vectors",
)
def delete_document(document_id: str) -> DeleteResponse:
    """Remove a document's record and all its vectors from ChromaDB."""
    record = get_document(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    deleted_count = delete_document_chunks(document_id)
    remove_document(document_id)

    logger.info("Deleted document '%s' (%d chunks)", document_id, deleted_count)
    return DeleteResponse(
        document_id=document_id,
        message=f"Deleted document and {deleted_count} associated chunks.",
    )
