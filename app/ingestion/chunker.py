"""Text chunking: split parsed pages into overlapping chunks with stable metadata."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from app.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str


def chunk_document(
    doc: ParsedDocument,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Split a parsed document into overlapping text chunks with stable IDs.

    Chunks stay within page boundaries so page metadata remains accurate.
    The overlap prevents context from being cut at chunk edges.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in doc.pages:
        page_chunks = _split_text(page.text, chunk_size, chunk_overlap)
        for text in page_chunks:
            if not text.strip():
                continue
            chunk_id = _make_chunk_id(document_id, page.page_number, chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=doc.filename,
                    page=page.page_number if doc.file_type == ".pdf" else None,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            chunk_index += 1

    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
        doc.filename,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size characters with overlap."""
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        parts.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return parts


def _make_chunk_id(document_id: str, page: int, index: int) -> str:
    """Generate a stable, collision-resistant chunk ID."""
    raw = f"{document_id}:p{page}:c{index}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"chunk_{digest}"
