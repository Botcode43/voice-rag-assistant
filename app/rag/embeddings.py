"""
app/rag/embeddings.py — Dense embedding generation via Ollama (nomic-embed-text).

All embedding calls are synchronous under the hood (Ollama HTTP call);
we expose an async wrapper for use in FastAPI/LangGraph async contexts.

Fix #1: A module-level persistent AsyncClient is used instead of creating
a new client per call. This enables TCP connection reuse (connection pooling)
to the Ollama server, saving ~50–150 ms of connection setup overhead per query.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Persistent HTTP client (connection-pooled, created once per process) ───────
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return (or lazily create) the module-level persistent AsyncClient."""
    global _http_client  # noqa: PLW0603
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


async def close_http_client() -> None:
    """Gracefully close the persistent HTTP client. Call from app lifespan shutdown."""
    global _http_client  # noqa: PLW0603
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("Embedding HTTP client closed")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate dense embeddings for a list of texts using Ollama's nomic-embed-text.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input text.

    Raises:
        RuntimeError: If the Ollama server is unreachable or returns an error.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    client = _get_http_client()
    try:
        response = await client.post(
            url,
            json={"model": settings.embedding_model, "input": texts},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Ollama embedding request failed: {exc}. "
            "Ensure Ollama is running and nomic-embed-text is pulled."
        ) from exc

    data = response.json()
    embeddings: list[list[float]] = data.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )

    logger.info("Generated %d dense embeddings", len(embeddings))
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0]
