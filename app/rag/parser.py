"""
app/rag/parser.py — PDF text extraction with PyMuPDF.
Returns a list of page-level text strings.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """
    Extract text from PDF bytes, one string per page.

    Args:
        pdf_bytes: Raw PDF file contents.

    Returns:
        A list of page-text strings (some may be empty for image-only pages).

    Raises:
        ValueError: If the bytes cannot be opened as a PDF.
        RuntimeError: If text extraction fails for all pages.
    """
    try:
        doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    pages: list[str] = []
    for page_num in range(len(doc)):
        try:
            page: fitz.Page = doc.load_page(page_num)
            text: str = page.get_text("text")  # type: ignore[arg-type]
            pages.append(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from page %d: %s", page_num, exc)
            pages.append("")

    doc.close()

    total_chars = sum(len(p) for p in pages)
    logger.info(
        "Parsed PDF: pages=%d total_chars=%d", len(pages), total_chars
    )

    if total_chars == 0:
        raise RuntimeError(
            "No text could be extracted from the PDF. "
            "The document may be image-only or encrypted."
        )

    return pages
