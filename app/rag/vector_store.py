"""
app/rag/vector_store.py — Qdrant vector store management.

Uses Qdrant in local/persistent mode (no Docker).
Each PDF gets its own collection (keyed by pdf_hash) so PDFs are isolated.

Both dense vectors (nomic-embed-text) and sparse vectors (FastEmbed BM25)
are stored per chunk, enabling server-side hybrid search via Qdrant's Query API.
"""

from __future__ import annotations

import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Sparse encoder (BM25 via FastEmbed) ───────────────────────────────────────
_sparse_encoder: SparseTextEmbedding | None = None


def _get_sparse_encoder() -> SparseTextEmbedding:
    global _sparse_encoder  # noqa: PLW0603
    if _sparse_encoder is None:
        logger.info("Loading FastEmbed BM25 sparse encoder …")
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("Sparse encoder ready")
    return _sparse_encoder


def encode_sparse(texts: list[str]) -> list[SparseVector]:
    """Return BM25 SparseVector objects for a list of texts."""
    encoder = _get_sparse_encoder()
    sparse_vectors: list[SparseVector] = []
    for result in encoder.embed(texts):
        sparse_vectors.append(
            SparseVector(
                indices=result.indices.tolist(),
                values=result.values.tolist(),
            )
        )
    return sparse_vectors


# ── Qdrant client singleton ────────────────────────────────────────────────────
_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """Return (or create) the module-level AsyncQdrantClient."""
    global _qdrant_client  # noqa: PLW0603
    if _qdrant_client is None:
        settings = get_settings()
        path = settings.qdrant_path
        logger.info("Initialising Qdrant client at path=%s", path)
        _qdrant_client = AsyncQdrantClient(path=path)
    return _qdrant_client


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def collection_name_for(pdf_hash: str) -> str:
    """Derive a Qdrant collection name from a PDF hash."""
    return f"rag_{pdf_hash[:16]}"


async def create_collection_if_missing(
    collection_name: str, embedding_dim: int
) -> bool:
    """
    Create a Qdrant collection with dense + sparse vector configs if it doesn't exist.

    Returns:
        True if collection was newly created, False if it already existed.
    """
    client = get_qdrant_client()
    existing = [c.name for c in (await client.get_collections()).collections]
    if collection_name in existing:
        logger.info("Collection already exists: %s", collection_name)
        return False

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            ),
        },
    )
    logger.info(
        "Created Qdrant collection: %s (dim=%d)", collection_name, embedding_dim
    )
    return True


async def upsert_chunks(
    collection_name: str,
    chunks: list[str],
    dense_vectors: list[list[float]],
) -> int:
    """
    Upsert text chunks with their dense + sparse vectors into Qdrant.

    Args:
        collection_name: Target Qdrant collection.
        chunks: Raw text strings.
        dense_vectors: Dense embedding per chunk (same order as *chunks*).

    Returns:
        Number of points upserted.
    """
    client = get_qdrant_client()
    sparse_vectors = encode_sparse(chunks)

    points: list[PointStruct] = []
    for i, (text, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: sparse,
                },
                payload={"text": text, "chunk_index": i},
            )
        )

    # Batch upsert in groups of 64 to avoid oversized payloads
    batch_size = 64
    for idx in range(0, len(points), batch_size):
        batch = points[idx : idx + batch_size]
        await client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "Upserted batch %d/%d (%d pts) into %s",
            idx // batch_size + 1,
            (len(points) + batch_size - 1) // batch_size,
            len(batch),
            collection_name,
        )

    return len(points)


async def collection_exists(collection_name: str) -> bool:
    """Return True if the collection exists in Qdrant (persistent check)."""
    client = get_qdrant_client()
    existing = [c.name for c in (await client.get_collections()).collections]
    return collection_name in existing


async def collection_count(collection_name: str) -> int:
    """Return the number of points in a collection."""
    client = get_qdrant_client()
    info = await client.get_collection(collection_name)
    return info.points_count or 0
