"""
app/services/pdf_service.py — Orchestrates PDF upload: parse → chunk → embed → store.

Two-layer deduplication:
  1. In-memory LRU pdf_cache (fast, resets on restart)
  2. Qdrant persistent storage check (survives restarts)

If the Qdrant collection for a PDF already exists and has points,
re-processing is skipped even after a server restart.
"""

from __future__ import annotations

from app.cache.cache import pdf_cache
from app.config import get_settings
from app.rag.chunker import chunk_pages
from app.rag.embeddings import embed_texts
from app.rag.parser import parse_pdf_bytes
from app.rag.vector_store import (
    collection_exists,
    collection_name_for,
    collection_count,
    create_collection_if_missing,
    upsert_chunks,
)
from app.utils.helpers import sha256_bytes
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def process_pdf(pdf_bytes: bytes, filename: str) -> tuple[str, str, bool, int]:
    """
    Process a PDF: parse → chunk → embed → store in Qdrant.

    Args:
        pdf_bytes: Raw PDF file bytes.
        filename: Original filename (for logging).

    Returns:
        (pdf_hash, collection_name, was_cached, num_chunks)
            - was_cached: True if the PDF hash was recognised and re-embedding was skipped.
            - num_chunks: Number of chunks in the collection (existing OR newly added).
    """
    settings = get_settings()
    pdf_hash = sha256_bytes(pdf_bytes)
    coll_name = collection_name_for(pdf_hash)

    logger.info(
        "Processing PDF: filename=%r hash=%s collection=%s",
        filename,
        pdf_hash[:16],
        coll_name,
    )

    # ── Layer 1: In-memory cache hit ───────────────────────────────────────────
    if pdf_cache.has(pdf_hash):
        existing_coll = pdf_cache.get(pdf_hash)
        count = await collection_count(existing_coll)
        logger.info(
            "PDF cache hit (memory): hash=%s collection=%s chunks=%d",
            pdf_hash[:16],
            existing_coll,
            count,
        )
        return pdf_hash, existing_coll, True, count

    # ── Layer 2: Qdrant persistent storage check (survives restarts) ───────────
    # The in-memory cache may be empty after a restart, but the Qdrant
    # collection may already exist on disk with all points intact.
    if await collection_exists(coll_name):
        count = await collection_count(coll_name)
        if count > 0:
            # Re-populate in-memory cache so subsequent requests are fast
            pdf_cache.set(pdf_hash, coll_name)
            logger.info(
                "PDF cache hit (Qdrant): hash=%s collection=%s chunks=%d",
                pdf_hash[:16],
                coll_name,
                count,
            )
            return pdf_hash, coll_name, True, count
        # Collection exists but is empty (partial write?) — fall through to reprocess

    # ── Parse ──────────────────────────────────────────────────────────────────
    logger.info("Parsing PDF pages …")
    pages = parse_pdf_bytes(pdf_bytes)

    # ── Chunk ──────────────────────────────────────────────────────────────────
    logger.info("Chunking text …")
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("PDF produced no usable text chunks.")

    # ── Embed ──────────────────────────────────────────────────────────────────
    logger.info("Embedding %d chunks …", len(chunks))
    dense_vectors = await embed_texts(chunks)

    # ── Store ──────────────────────────────────────────────────────────────────
    await create_collection_if_missing(coll_name, settings.embedding_dim)
    num_upserted = await upsert_chunks(coll_name, chunks, dense_vectors)

    # ── Cache the result ───────────────────────────────────────────────────────
    pdf_cache.set(pdf_hash, coll_name)
    logger.info(
        "PDF processed and cached: hash=%s chunks=%d", pdf_hash[:16], num_upserted
    )

    return pdf_hash, coll_name, False, num_upserted
