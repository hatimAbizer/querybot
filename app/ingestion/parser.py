"""Document parsing: extract text and per-page metadata from PDF, TXT, and Markdown files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def page_count(self) -> int | None:
        if self.file_type == ".pdf":
            return len(self.pages)
        return None


def parse_document(file_path: Path) -> ParsedDocument:
    """Parse a document file and return structured page-level content."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        return _parse_pdf(file_path)
    else:
        return _parse_text(file_path, suffix)


def _parse_pdf(file_path: Path) -> ParsedDocument:
    """Extract text from a PDF using pypdf, one page at a time."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF parsing. Install it with: pip install pypdf") from exc

    doc = ParsedDocument(filename=file_path.name, file_type=".pdf")
    try:
        reader = PdfReader(str(file_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = _normalize_whitespace(text)
            if text:
                doc.pages.append(ParsedPage(page_number=i + 1, text=text))
    except Exception as exc:
        logger.error("Failed to parse PDF '%s': %s", file_path.name, exc)
        raise

    logger.info("Parsed PDF '%s': %d pages with text", file_path.name, len(doc.pages))
    return doc


def _parse_text(file_path: Path, suffix: str) -> ParsedDocument:
    """Read a plain-text or Markdown file as a single page."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Failed to read file '%s': %s", file_path.name, exc)
        raise

    text = _normalize_whitespace(text)
    doc = ParsedDocument(
        filename=file_path.name,
        file_type=suffix,
        pages=[ParsedPage(page_number=1, text=text)],
    )
    logger.info("Parsed text file '%s': %d characters", file_path.name, len(text))
    return doc


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines; preserve paragraph structure."""
    import re

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
