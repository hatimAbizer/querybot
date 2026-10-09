"""Tests for context builder and prompt construction."""

from __future__ import annotations

from app.rag.context_builder import MAX_CONTEXT_CHARS, build_context, build_prompt
from app.schemas import ChunkResult


def _make_chunk(chunk_id: str = "c1", page: int | None = None, text: str = "Some text.") -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc_1",
        filename="test.txt",
        page=page,
        chunk_index=0,
        excerpt=text,
        distance=0.1,
    )


class TestBuildContext:
    def test_empty_chunks_returns_empty_string(self) -> None:
        assert build_context([]) == ""

    def test_single_chunk_included(self) -> None:
        chunk = _make_chunk(text="Important information here.")
        context = build_context([chunk])
        assert "Important information here." in context

    def test_includes_filename(self) -> None:
        chunk = _make_chunk()
        context = build_context([chunk])
        assert "test.txt" in context

    def test_includes_page_when_present(self) -> None:
        chunk = _make_chunk(page=5)
        context = build_context([chunk])
        assert "page 5" in context

    def test_no_page_label_when_none(self) -> None:
        chunk = _make_chunk(page=None)
        context = build_context([chunk])
        assert "page" not in context

    def test_context_bounded(self) -> None:
        # Generate chunks that would exceed the limit
        large_chunks = [_make_chunk(chunk_id=f"c{i}", text="x" * 1000) for i in range(20)]
        context = build_context(large_chunks)
        assert len(context) <= MAX_CONTEXT_CHARS + 500  # allow for labels

    def test_numbered_citations(self) -> None:
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(3)]
        context = build_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context


class TestBuildPrompt:
    def test_prompt_without_context_says_no_evidence(self) -> None:
        prompt = build_prompt("What is X?", context="")
        assert "no relevant passages" in prompt.lower() or "no relevant evidence" in prompt.lower()

    def test_prompt_with_context_contains_question(self) -> None:
        prompt = build_prompt("What is the revenue?", context="[1]\nRevenue was $100M.")
        assert "What is the revenue?" in prompt

    def test_prompt_with_context_contains_context(self) -> None:
        ctx = "[1]\nRevenue was $100M."
        prompt = build_prompt("Revenue?", context=ctx)
        assert ctx in prompt
