"""
app/rag/embeddings.py — Dense embedding generation via Ollama (nomic-embed-text).

All embedding calls are synchronous under the hood (Ollama HTTP call);
we expose an async wrapper for use in FastAPI/LangGraph async contexts.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


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

    async with httpx.AsyncClient(timeout=120.0) as client:
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
