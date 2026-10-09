"""Query API: semantic search and RAG ask endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.rag.rag_service import answer_question
from app.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from app.services.retrieval_service import search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve relevant chunks without generation",
)
def semantic_search(request: SearchRequest) -> SearchResponse:
    """Embed the query and return the top-k most relevant document chunks."""
    try:
        results = search(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc

    return SearchResponse(query=request.query, results=results, total=len(results))


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Generate an evidence-grounded answer",
)
def ask(request: AskRequest) -> AskResponse:
    """Retrieve relevant evidence and generate a grounded answer via the local LLM."""
    try:
        response = answer_question(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except Exception as exc:
        logger.exception("Ask failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ask failed: {exc}",
        ) from exc

    return response
