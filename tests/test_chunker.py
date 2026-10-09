"""Tests for the text chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import _make_chunk_id, _split_text, chunk_document
from app.ingestion.parser import ParsedDocument, ParsedPage


def _make_doc(text: str, suffix: str = ".txt") -> ParsedDocument:
    return ParsedDocument(
        filename="test.txt",
        file_type=suffix,
        pages=[ParsedPage(page_number=1, text=text)],
    )


class TestSplitText:
    def test_short_text_single_chunk(self) -> None:
        parts = _split_text("hello", chunk_size=100, chunk_overlap=10)
        assert parts == ["hello"]

    def test_long_text_multiple_chunks(self) -> None:
        text = "a" * 300
        parts = _split_text(text, chunk_size=100, chunk_overlap=20)
        assert len(parts) > 1
        # Every part must be at most chunk_size characters
        assert all(len(p) <= 100 for p in parts)

    def test_overlap_is_applied(self) -> None:
        text = "abcdefghij"  # 10 chars
        # chunk_size=6, overlap=2 → step=4
        # chunks: [0:6], [4:10]
        parts = _split_text(text, chunk_size=6, chunk_overlap=2)
        assert parts[0] == "abcdef"
        assert parts[1] == "efghij"


class TestChunkDocument:
    def test_basic_chunking(self) -> None:
        doc = _make_doc("word " * 200)  # 1000 chars
        chunks = chunk_document(doc, document_id="doc_abc", chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(c.document_id == "doc_abc" for c in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        doc = _make_doc("word " * 300)
        chunks = chunk_document(doc, document_id="doc_xyz", chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_sequential(self) -> None:
        doc = _make_doc("word " * 200)
        chunks = chunk_document(doc, document_id="doc_seq", chunk_size=100, chunk_overlap=20)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_txt_page_is_none(self) -> None:
        doc = _make_doc("word " * 100, suffix=".txt")
        chunks = chunk_document(doc, document_id="doc_txt", chunk_size=50, chunk_overlap=10)
        assert all(c.page is None for c in chunks)

    def test_pdf_page_is_set(self) -> None:
        doc = ParsedDocument(
            filename="test.pdf",
            file_type=".pdf",
            pages=[ParsedPage(page_number=3, text="word " * 100)],
        )
        chunks = chunk_document(doc, document_id="doc_pdf", chunk_size=50, chunk_overlap=10)
        assert all(c.page == 3 for c in chunks)

    def test_invalid_chunk_size_raises(self) -> None:
        doc = _make_doc("text")
        with pytest.raises(ValueError):
            chunk_document(doc, document_id="d", chunk_size=0, chunk_overlap=0)

    def test_invalid_overlap_raises(self) -> None:
        doc = _make_doc("text")
        with pytest.raises(ValueError):
            chunk_document(doc, document_id="d", chunk_size=100, chunk_overlap=100)


class TestMakeChunkId:
    def test_deterministic(self) -> None:
        id1 = _make_chunk_id("doc_a", 1, 0)
        id2 = _make_chunk_id("doc_a", 1, 0)
        assert id1 == id2

    def test_different_inputs_differ(self) -> None:
        id1 = _make_chunk_id("doc_a", 1, 0)
        id2 = _make_chunk_id("doc_b", 1, 0)
        assert id1 != id2

    def test_prefix(self) -> None:
        cid = _make_chunk_id("doc_a", 1, 0)
        assert cid.startswith("chunk_")
