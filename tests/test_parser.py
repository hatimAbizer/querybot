"""Tests for document parsing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion.parser import SUPPORTED_EXTENSIONS, _normalize_whitespace, parse_document


def _write_tmp(suffix: str, content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


class TestNormalizeWhitespace:
    def test_collapses_blank_lines(self) -> None:
        result = _normalize_whitespace("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_collapses_spaces(self) -> None:
        result = _normalize_whitespace("hello   world")
        assert result == "hello world"

    def test_strips_edges(self) -> None:
        result = _normalize_whitespace("  hello  ")
        assert result == "hello"


class TestParseTxt:
    def test_parse_txt_returns_single_page(self) -> None:
        path = _write_tmp(".txt", "Hello world\nThis is a test.")
        try:
            doc = parse_document(path)
            assert doc.file_type == ".txt"
            assert len(doc.pages) == 1
            assert "Hello world" in doc.pages[0].text
        finally:
            path.unlink(missing_ok=True)

    def test_parse_txt_page_count_is_none(self) -> None:
        path = _write_tmp(".txt", "Some content")
        try:
            doc = parse_document(path)
            assert doc.page_count is None
        finally:
            path.unlink(missing_ok=True)


class TestParseMarkdown:
    def test_parse_md(self) -> None:
        content = "# Heading\n\nSome paragraph text."
        path = _write_tmp(".md", content)
        try:
            doc = parse_document(path)
            assert doc.file_type == ".md"
            assert "Heading" in doc.full_text
        finally:
            path.unlink(missing_ok=True)


class TestUnsupportedExtension:
    def test_raises_for_unsupported_type(self) -> None:
        path = _write_tmp(".docx", "content")
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                parse_document(path)
        finally:
            path.unlink(missing_ok=True)


class TestSupportedExtensions:
    def test_supported_extensions_set(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
