"""
app/services/streaming_service.py — SSE event stream generator.

This is the core of the streaming pipeline:
1. Run the LangGraph (injection → decomposition → retrieval → grounding)
2. If grounded → start Gemini stream → per-sentence TTS → emit SSE events
3. LLM response caching: store successful answers, replay as stream on cache hit

SSE event format (newline-delimited JSON in 'data:' field):
  data: {"type": "text", "content": "..."}
  data: {"type": "audio", "audio_b64": "..."}
  data: {"type": "done"}
  data: {"type": "error", "error": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator

from app.cache.cache import llm_cache
from app.config import get_settings
from app.graph.graph import rag_graph
from app.graph.state import RAGState
from app.llm.gemini import stream_generate_v2
from app.llm.prompts import FALLBACK_ANSWER, build_rag_prompt
from app.memory.memory import session_memory
from app.rag.vector_store import collection_name_for
from app.services.tts_service import stream_tts_for_text_stream
from app.utils.helpers import make_llm_cache_key
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


async def run_rag_stream(
    session_id: str,
    pdf_hash: str,
    query: str,
) -> AsyncIterator[str]:
    """
    Full pipeline: graph → optional LLM stream → SSE events.

    Yields SSE-formatted strings to be consumed by a FastAPI StreamingResponse.
    """
    settings = get_settings()
    collection_name = collection_name_for(pdf_hash)

    # ── Build initial state ────────────────────────────────────────────────────
    initial_state: RAGState = {
        "session_id": session_id,
        "pdf_hash": pdf_hash,
        "original_query": query,
        "collection_name": collection_name,
    }

    # ── Run graph (injection → decomposition → retrieval → grounding) ──────────
    try:
        final_state: RAGState = await rag_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution error: %s", exc)
        yield _sse({"type": "error", "error": str(exc)})
        return

    # ── Short-circuit paths (injection or not-grounded) ────────────────────────
    final_answer: str | None = final_state.get("final_answer")
    if final_answer:
        # Emit text
        yield _sse({"type": "text", "content": final_answer})
        # Emit TTS for the fallback/rejection message
        try:
            audio_bytes = await synthesise_text(final_answer)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            yield _sse({"type": "audio", "audio_b64": audio_b64})
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS failed for fallback message: %s", exc)
        yield _sse({"type": "done"})
        return

    # ── Grounded path: check LLM cache first ──────────────────────────────────
    cache_key = make_llm_cache_key(pdf_hash, query)
    cached_answer: str | None = llm_cache.get(cache_key)

    if cached_answer:
        logger.info("LLM cache hit for key=%s", cache_key[:12])

        # Fix #8: removed asyncio.sleep(0.02) — was adding ~400–800 ms of fake latency.
        # Fix #6: use stream_tts_for_text_stream so TTS runs sentence-by-sentence in
        #          parallel (same strategy as the live path), instead of blocking on the
        #          full answer string — saves ~1–3 s before first audio byte on cache hits.
        async def _cached_text_stream() -> AsyncIterator[str]:
            words = cached_answer.split()
            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk

        try:
            async for text_token, audio_bytes in stream_tts_for_text_stream(_cached_text_stream()):
                if text_token:
                    yield _sse({"type": "text", "content": text_token})
                if audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    yield _sse({"type": "audio", "audio_b64": audio_b64})
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS failed for cached answer: %s", exc)

        yield _sse({"type": "done"})

        # Store in session memory
        session_memory.add_turn(session_id, "user", query)
        session_memory.add_turn(session_id, "assistant", cached_answer)
        return

    # ── Build prompt and stream from Gemini ────────────────────────────────────
    context_text = final_state.get("context_text", "")
    resolved_query = final_state.get("resolved_query") or query
    conversation_history = session_memory.format_for_context(session_id)

    prompt = build_rag_prompt(
        context=context_text,
        query=resolved_query,
        conversation_history=conversation_history,
    )

    accumulated_answer = ""

    try:
        async def _text_stream() -> AsyncIterator[str]:
            async for token in stream_generate_v2(prompt):
                yield token

        async for text_token, audio_bytes in stream_tts_for_text_stream(_text_stream()):
            if text_token:
                accumulated_answer += text_token
                yield _sse({"type": "text", "content": text_token})

            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode()
                yield _sse({"type": "audio", "audio_b64": audio_b64})

    except Exception as exc:
        logger.exception("Streaming generation error: %s", exc)
        yield _sse({"type": "error", "error": str(exc)})
        return

    yield _sse({"type": "done"})

    # ── Cache successful grounded answer ───────────────────────────────────────
    if accumulated_answer and accumulated_answer != FALLBACK_ANSWER:
        llm_cache.set(cache_key, accumulated_answer)
        logger.info("Cached LLM answer (key=%s)", cache_key[:12])

    # ── Update session memory ──────────────────────────────────────────────────
    session_memory.add_turn(session_id, "user", query)
    session_memory.add_turn(session_id, "assistant", accumulated_answer)
