"""
app/rag/chunker.py — Token-aware text chunking with RecursiveCharacterTextSplitter.

Uses tiktoken (cl100k_base) to measure token lengths so chunk_size=700 means
~700 tokens, not characters.
"""

from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Single tokenizer instance for the process
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Fix #4: splitter built once — chunk_size/overlap come from settings (a cached
# singleton) so the config never changes. No reason to reconstruct per upload.
_splitter: RecursiveCharacterTextSplitter | None = None


def _token_len(text: str) -> int:
    """Token-aware length function for the text splitter."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter  # noqa: PLW0603
    if _splitter is None:
        settings = get_settings()
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=_token_len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter


def chunk_pages(pages: list[str]) -> list[str]:
    """
    Join all pages into a single document, then split into overlapping chunks.

    Args:
        pages: List of per-page text strings from the PDF parser.

    Returns:
        List of text chunk strings.
    """
    settings = get_settings()
    full_text = "\n\n".join(p for p in pages if p.strip())

    splitter = _get_splitter()
    chunks: list[str] = splitter.split_text(full_text)
    logger.info(
        "Chunked document: total_chunks=%d chunk_size=%d chunk_overlap=%d",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
