"""
app/graph/graph.py — LangGraph pipeline construction and compilation.

Pipeline:
  START
  → injection_check
  → [conditional] query_decomposition | END
  → retrieval
  → grounding_check
  → [conditional] generation (streaming, handled by streaming_service) | END
  → END

Note: The 'generation' node is NOT inside the graph because it is a streaming
operation — we run the graph up to the grounding_check to resolve whether
we should generate, then hand off to streaming_service for the actual stream.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_after_grounding_check, route_after_injection_check
from app.graph.nodes import (
    grounding_check_node,
    injection_check_node,
    query_decomposition_node,
    retrieval_node,
)
from app.graph.state import RAGState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    """Build and compile the LangGraph pipeline (up to generation decision)."""
    graph = StateGraph(RAGState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("injection_check", injection_check_node)
    graph.add_node("query_decomposition", query_decomposition_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grounding_check", grounding_check_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "injection_check")

    graph.add_conditional_edges(
        "injection_check",
        route_after_injection_check,
        {
            "end": END,
            "query_decomposition": "query_decomposition",
        },
    )

    graph.add_edge("query_decomposition", "retrieval")
    graph.add_edge("retrieval", "grounding_check")

    graph.add_conditional_edges(
        "grounding_check",
        route_after_grounding_check,
        {
            "end": END,
            "generation": END,   # We exit graph here; streaming_service takes over
        },
    )

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


# Module-level compiled graph singleton
rag_graph = build_graph()
