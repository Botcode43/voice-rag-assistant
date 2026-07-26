"""
app/graph/edges.py — Conditional edge functions for the LangGraph.

These functions inspect state and return the name of the next node to route to.
"""

from __future__ import annotations

from app.graph.state import RAGState


def route_after_injection_check(state: RAGState) -> str:
    """
    After injection_check:
    - If injection detected → 'end' (short-circuit)
    - Otherwise → 'query_decomposition'
    """
    if state.get("injection_detected"):
        return "end"
    return "query_decomposition"


def route_after_grounding_check(state: RAGState) -> str:
    """
    After grounding_check:
    - If not grounded → 'end' (return fallback, skip LLM)
    - If grounded → 'generation'
    """
    if not state.get("is_grounded"):
        return "end"
    return "generation"
