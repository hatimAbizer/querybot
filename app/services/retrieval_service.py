"""Retrieval service: embed query, query ChromaDB, filter by relevance threshold."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.retrieval.embedder import embed_query
from app.retrieval.vector_store import retrieve_chunks
from app.schemas import ChunkResult

logger = logging.getLogger(__name__)


def search(
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[ChunkResult]:
    """Embed query, retrieve top-k chunks, and filter by relevance threshold.

    Returns chunks sorted by ascending distance (most similar first).
    Chunks whose cosine distance exceeds the relevance threshold are excluded.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k

    query_embedding = embed_query(query)
    raw = retrieve_chunks(query_embedding, top_k=k, document_ids=document_ids)

    results: list[ChunkResult] = []
    for r in raw:
        distance = float(r["distance"])
        if distance > settings.relevance_threshold:
            # Skip chunks that are too dissimilar
            continue
        meta = r["metadata"]
        page_val = meta.get("page", -1)
        results.append(
            ChunkResult(
                chunk_id=r["chunk_id"],
                document_id=meta["document_id"],
                filename=meta["filename"],
                page=page_val if page_val != -1 else None,
                chunk_index=int(meta.get("chunk_index", 0)),
                excerpt=r["text"],
                distance=distance,
            )
        )

    logger.info(
        "Search '%s…': %d/%d chunks passed threshold %.2f",
        query[:40],
        len(results),
        len(raw),
        settings.relevance_threshold,
    )
    return results
