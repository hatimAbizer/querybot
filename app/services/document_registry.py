"""In-memory document registry: stores document records by document_id.

For Milestone 1 this is a simple in-process dict. It is intentionally not
persisted to disk in this milestone — on restart the API will show no
documents even though ChromaDB still holds their chunks. The registry is
populated each time via the upload endpoint. Persistence is a Milestone 2
concern.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from app.schemas import DocumentRecord

logger = logging.getLogger(__name__)

_registry: dict[str, DocumentRecord] = {}


def generate_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


def register_document(
    document_id: str,
    filename: str,
    file_type: str,
    chunk_count: int,
    page_count: int | None,
) -> DocumentRecord:
    record = DocumentRecord(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        chunk_count=chunk_count,
        page_count=page_count,
        created_at=datetime.now(UTC).isoformat(),
    )
    _registry[document_id] = record
    logger.info("Registered document '%s' (%s)", filename, document_id)
    return record


def get_document(document_id: str) -> DocumentRecord | None:
    return _registry.get(document_id)


def list_documents() -> list[DocumentRecord]:
    return list(_registry.values())


def remove_document(document_id: str) -> bool:
    """Remove a document record. Returns True if it existed."""
    if document_id in _registry:
        del _registry[document_id]
        logger.info("Removed document record '%s'", document_id)
        return True
    return False
