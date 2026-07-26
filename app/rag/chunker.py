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


def _token_len(text: str) -> int:
    """Token-aware length function for the text splitter."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[str] = splitter.split_text(full_text)
    logger.info(
        "Chunked document: total_chunks=%d chunk_size=%d chunk_overlap=%d",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
