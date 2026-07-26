"""
app/services/rag_service.py — Thin orchestration layer for the query pipeline.

This module is the single entry point called by API endpoints.
It delegates to streaming_service.run_rag_stream for the full pipeline.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.cache.cache import pdf_cache
from app.rag.vector_store import collection_name_for
from app.services.streaming_service import run_rag_stream
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def handle_query(
    session_id: str,
    pdf_hash: str,
    query: str,
) -> AsyncIterator[str]:
    """
    Run the full RAG pipeline for a text query and return an SSE event stream.

    Args:
        session_id: Unique session identifier.
        pdf_hash: SHA-256 hash of the previously uploaded PDF.
        query: User's question text.

    Returns:
        Async iterator of SSE-formatted strings.

    Raises:
        ValueError: If the pdf_hash is not found in the PDF cache.
    """
    if not pdf_cache.has(pdf_hash):
        raise ValueError(
            f"PDF with hash {pdf_hash[:16]}… has not been uploaded. "
            "Please upload a PDF first."
        )

    logger.info(
        "RAG query: session=%s pdf_hash=%s query=%r",
        session_id,
        pdf_hash[:16],
        query[:80],
    )

    return run_rag_stream(session_id=session_id, pdf_hash=pdf_hash, query=query)
