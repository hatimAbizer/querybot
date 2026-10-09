"""ChromaDB vector store: upsert, retrieve, and delete document chunks."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "querybot_chunks"


@lru_cache(maxsize=1)
def _get_collection():
    """Return (and cache) the persistent ChromaDB collection."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("chromadb is required. Install it with: pip install chromadb") from exc

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready at '%s'",
        COLLECTION_NAME,
        settings.chroma_path,
    )
    return collection


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert chunks and their embeddings into ChromaDB."""
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings lists must have the same length")

    collection = _get_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "filename": c.filename,
            "page": c.page if c.page is not None else -1,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into ChromaDB", len(chunks))


def retrieve_chunks(
    query_embedding: list[float],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Query ChromaDB and return raw results dicts."""
    collection = _get_collection()

    where: dict | None = None
    if document_ids:
        if len(document_ids) == 1:
            where = {"document_id": document_ids[0]}
        else:
            where = {"document_id": {"$in": document_ids}}

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten from batched format (batch size = 1)
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "chunk_id": cid,
            "text": doc,
            "metadata": meta,
            "distance": dist,
        }
        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=False)
    ]


def delete_document_chunks(document_id: str) -> int:
    """Delete all chunks belonging to a document. Returns number of deleted chunks."""
    collection = _get_collection()

    # Get IDs of chunks to delete
    results = collection.get(
        where={"document_id": document_id},
        include=[],
    )
    ids_to_delete = results.get("ids", [])
    count = len(ids_to_delete)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for document '%s'", count, document_id)
    else:
        logger.warning("No chunks found for document '%s'", document_id)

    return count


def get_collection_count() -> int:
    """Return total number of chunks in the collection."""
    return _get_collection().count()
