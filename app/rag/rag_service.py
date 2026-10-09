"""RAG service: orchestrate retrieval → context → generation → citation validation."""

from __future__ import annotations

import logging
import uuid

from app.rag.context_builder import build_context, build_prompt
from app.rag.generator import generate
from app.schemas import AskResponse, ChunkResult, SourceReference
from app.services.retrieval_service import search

logger = logging.getLogger(__name__)

# If fewer than this many chunks pass the threshold we treat evidence as insufficient
MIN_EVIDENCE_CHUNKS = 1


def answer_question(
    question: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> AskResponse:
    """Full RAG pipeline: retrieve evidence, generate grounded answer, validate citations.

    Returns an AskResponse with answer, validated sources, and evidence status.
    If evidence is insufficient the answer says so and no sources are included.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info("[%s] Question: %s…", request_id, question[:60])

    # --- Retrieval ---
    chunks: list[ChunkResult] = search(question, top_k=top_k, document_ids=document_ids)
    insufficient = len(chunks) < MIN_EVIDENCE_CHUNKS

    # --- Context and prompt ---
    context = build_context(chunks)
    prompt = build_prompt(question, context)

    # --- Generation ---
    try:
        raw_answer = generate(prompt)
    except RuntimeError as exc:
        logger.error("[%s] Generation failed: %s", request_id, exc)
        return AskResponse(
            answer=(
                "The answer generation service is currently unavailable. "
                f"({exc})"
            ),
            sources=[],
            insufficient_evidence=True,
            request_id=request_id,
        )

    # --- Citation validation ---
    # Only include sources whose chunk_id was actually retrieved; never invent them.
    retrieved_ids = {c.chunk_id for c in chunks}
    validated_sources: list[SourceReference] = []
    for chunk in chunks:
        if chunk.chunk_id in retrieved_ids:
            validated_sources.append(
                SourceReference(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    excerpt=chunk.excerpt,
                )
            )

    logger.info(
        "[%s] Done — %d sources, insufficient=%s",
        request_id,
        len(validated_sources),
        insufficient,
    )
    return AskResponse(
        answer=raw_answer,
        sources=validated_sources if not insufficient else [],
        insufficient_evidence=insufficient,
        request_id=request_id,
    )
