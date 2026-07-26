"""
app/llm/prompts.py — System and user prompt templates.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an AI assistant that answers questions ONLY from the retrieved document context.

Rules:
- Never use outside knowledge.
- Never guess or fabricate information.
- If the answer cannot be found in the context, reply exactly:
  "I couldn't find that information in the uploaded document."
- Keep answers concise and factual."""

FALLBACK_ANSWER = "I couldn't find that information in the uploaded document."


def build_rag_prompt(
    context: str,
    query: str,
    conversation_history: str = "",
) -> str:
    """
    Construct the full user-turn prompt for Gemini.

    Args:
        context: Retrieved document chunks joined into a single string.
        query: The (decomposed) user question.
        conversation_history: Optional formatted chat history for follow-up context.

    Returns:
        The user-turn prompt string.
    """
    history_block = ""
    if conversation_history:
        history_block = f"\n\n--- Conversation History ---\n{conversation_history}\n"

    return (
        f"--- Retrieved Document Context ---\n{context}\n"
        f"{history_block}"
        f"\n--- Question ---\n{query}\n"
        f"\nAnswer using ONLY the context above:"
    )
