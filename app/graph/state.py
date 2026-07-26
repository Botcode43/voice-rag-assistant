"""
app/graph/state.py — LangGraph state definition for the RAG pipeline.

All nodes read from and write to this TypedDict.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):
    """Mutable state that flows through the LangGraph nodes."""

    # ── Inputs ─────────────────────────────────────────────────────────────────
    session_id: str                         # Unique session identifier
    pdf_hash: str                           # SHA-256 hash of the uploaded PDF
    original_query: str                     # Raw question from the user
    collection_name: str                    # Qdrant collection for this PDF

    # ── Guardrails ─────────────────────────────────────────────────────────────
    injection_detected: bool                # True → short-circuit immediately

    # ── Query decomposition ────────────────────────────────────────────────────
    resolved_query: str                     # After resolving follow-ups via memory

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_chunks: list[dict[str, Any]]  # [{text, score, chunk_index}, ...]
    top_score: float                        # Highest relevance score from retrieval
    context_text: str                       # Joined chunk texts for LLM prompt

    # ── Grounding check ────────────────────────────────────────────────────────
    is_grounded: bool                       # True → send to LLM; False → fallback

    # ── Generation ─────────────────────────────────────────────────────────────
    answer_text: str                        # Full accumulated answer
    from_cache: bool                        # True if answer came from LLM cache

    # ── Control flow ──────────────────────────────────────────────────────────
    error: Optional[str]                    # Set if a recoverable error occurred
    final_answer: str                       # Final answer text sent to the client
