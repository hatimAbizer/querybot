"""Ingestion service: orchestrate parse → chunk → embed → index pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import SUPPORTED_EXTENSIONS, parse_document
from app.retrieval.embedder import embed_texts
from app.retrieval.vector_store import upsert_chunks
from app.schemas import UploadResponse
from app.services.document_registry import generate_document_id, register_document

logger = logging.getLogger(__name__)


def ingest_file(filename: str, content: bytes) -> UploadResponse:
    """Validate, parse, chunk, embed, and index an uploaded file.

    Args:
        filename: Original filename (used for type detection and metadata).
        content: Raw file bytes.

    Returns:
        UploadResponse with document metadata and chunk count.

    Raises:
        ValueError: For unsupported file types or size limit violations.
    """
    settings = get_settings()
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if len(content) > settings.max_file_size_bytes:
        raise ValueError(
            f"File exceeds the {settings.max_file_size_mb} MB size limit."
        )

    document_id = generate_document_id()
    logger.info("Starting ingestion: '%s' → %s", filename, document_id)

    # Write to a temp file so parsers can use file-system APIs
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parsed = parse_document(tmp_path)
        parsed.filename = filename  # restore original name
    finally:
        tmp_path.unlink(missing_ok=True)

    chunks = chunk_document(
        parsed,
        document_id=document_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(chunks, embeddings)

    record = register_document(
        document_id=document_id,
        filename=filename,
        file_type=suffix,
        chunk_count=len(chunks),
        page_count=parsed.page_count,
    )

    logger.info("Ingestion complete: '%s' → %d chunks", filename, len(chunks))
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
        page_count=record.page_count,
        message=f"Successfully indexed {len(chunks)} chunks.",
    )
