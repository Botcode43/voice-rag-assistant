"""
app/rag/retriever.py — Hybrid retrieval using Qdrant's Query API (RRF fusion).

Dense + sparse query vectors are sent in a single request to Qdrant which fuses
results server-side via Reciprocal Rank Fusion.  No BM25 index or manual score
merging in application code.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from app.config import get_settings
from app.rag.vector_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    encode_sparse,
    get_qdrant_client,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_index: int


async def hybrid_retrieve(
    collection_name: str,
    query: str,
    dense_vector: list[float],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Run Qdrant hybrid search (dense + sparse, RRF-fused server-side).

    Args:
        collection_name: Qdrant collection to search.
        query: Raw query text (used to generate sparse vector).
        dense_vector: Pre-computed dense embedding for the query.
        top_k: Number of results to return (defaults to settings.retrieval_top_k).

    Returns:
        List of RetrievedChunk sorted best-first by RRF score.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    client: AsyncQdrantClient = get_qdrant_client()

    # Sparse encoding for the query
    sparse_vecs = encode_sparse([query])
    sparse_vec: SparseVector = sparse_vecs[0]

    # Qdrant Query API — Prefetch dense + sparse then fuse with RRF
    results = await client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vector,
                limit=k * 2,
                using=DENSE_VECTOR_NAME,
                score_threshold=settings.dense_similarity_threshold,
            ),
            Prefetch(
                query=sparse_vec,
                limit=k * 2,
                using=SPARSE_VECTOR_NAME,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    chunks: list[RetrievedChunk] = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                score=point.score,
                chunk_index=payload.get("chunk_index", -1),
            )
        )

    logger.info(
        "Retrieved %d chunks from %s (top score=%.4f)",
        len(chunks),
        collection_name,
        chunks[0].score if chunks else 0.0,
    )
    return chunks
