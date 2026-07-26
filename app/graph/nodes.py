"""
app/graph/nodes.py — LangGraph node implementations.

Each node is an async function that takes RAGState and returns a partial state
dict with the fields it updates.

Node order: injection_check → query_decomposition → retrieval → grounding_check
            → (generation handled separately via streaming service)
"""

from __future__ import annotations

import re
from typing import Any

from app.config import get_settings
from app.guardrails.prompt_guard import INJECTION_REJECTION_MESSAGE, is_injection
from app.llm.prompts import FALLBACK_ANSWER
from app.memory.memory import session_memory
from app.rag.embeddings import embed_query
from app.rag.retriever import hybrid_retrieve
from app.graph.state import RAGState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Node 1: Prompt Injection Check ────────────────────────────────────────────

async def injection_check_node(state: RAGState) -> dict[str, Any]:
    """
    Fast regex check for prompt injection patterns.
    Sets injection_detected=True and final_answer if blocked.
    """
    query = state.get("original_query", "")
    logger.info("Node: injection_check (session=%s)", state.get("session_id"))

    if is_injection(query):
        return {
            "injection_detected": True,
            "final_answer": INJECTION_REJECTION_MESSAGE,
        }

    return {"injection_detected": False}


# ── Node 2: Query Decomposition ───────────────────────────────────────────────

async def query_decomposition_node(state: RAGState) -> dict[str, Any]:
    """
    Resolve follow-up questions using session memory.

    For now this is a lightweight rule-based approach:
    - If the query contains anaphoric pronouns ('it', 'he', 'she', 'they', 'this',
      'that', 'the author', 'the title', etc.) without explicit subjects,
      prepend the last assistant turn's core entity for disambiguation.

    No LLM call is made here — this keeps latency near-zero.
    """
    session_id = state.get("session_id", "")
    original = state.get("original_query", "")
    logger.info("Node: query_decomposition (session=%s)", session_id)

    history = session_memory.get_history(session_id)

    # Simple resolution: build a combined query that includes recent context
    # if the current question looks like a follow-up (short, starts with pronoun/article)
    resolved = original
    if history:
        last_user_turn = next(
            (h["content"] for h in reversed(history) if h["role"] == "user"),
            None,
        )
        # Heuristic: short queries (<= 8 words) that start with common anaphora
        words = original.strip().split()
        anaphora = {"it", "he", "she", "they", "this", "that", "its", "their", "the", "when", "where", "why", "how"}
        if len(words) <= 8 and words[0].lower() in anaphora and last_user_turn:
            resolved = f"{last_user_turn} — {original}"
            logger.info(
                "Query resolved (follow-up): %r → %r", original, resolved
            )

    return {"resolved_query": resolved}


# ── Node 3: Retrieval ─────────────────────────────────────────────────────────

async def retrieval_node(state: RAGState) -> dict[str, Any]:
    """
    Embed the resolved query and run Qdrant hybrid search (dense + sparse, RRF).
    """
    session_id = state.get("session_id", "")
    query = state.get("resolved_query") or state.get("original_query", "")
    collection_name = state.get("collection_name", "")
    settings = get_settings()

    logger.info("Node: retrieval (session=%s collection=%s)", session_id, collection_name)

    dense_vec = await embed_query(query)
    chunks = await hybrid_retrieve(
        collection_name=collection_name,
        query=query,
        dense_vector=dense_vec,
        top_k=settings.retrieval_top_k,
    )

    retrieved = [
        {"text": c.text, "score": c.score, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    top_score = chunks[0].score if chunks else 0.0
    context_text = "\n\n---\n\n".join(c.text for c in chunks)

    logger.info(
        "Retrieval complete: chunks=%d top_score=%.4f", len(retrieved), top_score
    )

    return {
        "retrieved_chunks": retrieved,
        "top_score": top_score,
        "context_text": context_text,
    }


# ── Node 4: Context / Grounding Check ────────────────────────────────────────

async def grounding_check_node(state: RAGState) -> dict[str, Any]:
    """
    Compare top similarity score against threshold.
    No LLM call — purely numeric comparison.
    """
    settings = get_settings()
    top_score = state.get("top_score", 0.0)
    session_id = state.get("session_id", "")

    logger.info(
        "Node: grounding_check (session=%s top_score=%.4f threshold=%.4f)",
        session_id,
        top_score,
        settings.grounding_threshold,
    )

    is_grounded = top_score >= settings.grounding_threshold

    if not is_grounded:
        logger.info(
            "Not grounded — returning fallback (score=%.4f < threshold=%.4f)",
            top_score,
            settings.grounding_threshold,
        )
        return {
            "is_grounded": False,
            "final_answer": FALLBACK_ANSWER,
        }

    return {"is_grounded": True}
