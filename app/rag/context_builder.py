"""RAG context builder: assemble a bounded, citation-safe context string from retrieved chunks."""

from __future__ import annotations

from app.schemas import ChunkResult

# Max characters of context sent to the LLM. Kept small to reduce generation
# time on CPU/partial-GPU machines; 3000 chars ≈ 750 tokens of context.
MAX_CONTEXT_CHARS = 3000


def build_context(chunks: list[ChunkResult]) -> str:
    """Assemble numbered context passages from retrieved chunks.

    Each passage is prefixed with a citation tag like [1] so the LLM can
    reference it. The total length is bounded by MAX_CONTEXT_CHARS.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}] (file: {chunk.filename}"
        if chunk.page is not None:
            label += f", page {chunk.page}"
        label += ")"
        passage = f"{label}\n{chunk.excerpt}"
        passage_len = len(passage)

        if total + passage_len > MAX_CONTEXT_CHARS and parts:
            # Always include at least the first chunk
            break
        parts.append(passage)
        total += passage_len

    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Build the final LLM prompt from the question and assembled context."""
    if not context:
        return (
            "You are a document assistant. You have no relevant passages available.\n"
            "The user asked: " + question + "\n\n"
            "Since there is no relevant evidence in the indexed documents, "
            "clearly state that you cannot answer this question from the available documents."
        )

    return (
        "You are a document assistant. Answer the user's question using ONLY the "
        "numbered passages provided below. Do not use outside knowledge.\n\n"
        "Rules:\n"
        "- Cite the passage number(s) that support each claim, e.g. [1] or [2, 3].\n"
        "- If the passages do not contain sufficient information to answer, say so explicitly.\n"
        "- Never fabricate information not present in the passages.\n"
        "- Keep the answer concise and grounded.\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
