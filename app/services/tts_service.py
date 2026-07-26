"""
app/services/tts_service.py — Sentence-level TTS coordination.

Strategy:
  Phase 1 — stream all text tokens immediately, launch a TTS asyncio.Task
             for each completed sentence (parallel synthesis in background).
  Phase 2 — after all text is streamed, yield audio chunks IN ORDER by
             awaiting each sentence's Task sequentially.

This approach:
  - Yields text tokens with zero delay (low TTFT).
  - Synthesises all sentences in parallel while text is streaming.
  - Sends audio to the client strictly one-at-a-time (no overlapping voices).
  - Never deadlocks because there is no shared Queue or asyncio.wait racing.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from app.speech.edge_tts import synthesise_sentence
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Sentence boundary: after . ! ? followed by whitespace or end-of-string
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract all complete sentences from *buffer* and return the remainder.

    Returns:
        (complete_sentences, remaining_buffer)
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return complete, remainder


async def stream_tts_for_text_stream(
    text_stream: AsyncIterator[str],
) -> AsyncIterator[tuple[str, bytes | None]]:
    """
    Consume a text token stream and yield (text_chunk, audio_bytes_or_None).

    Text tokens are yielded immediately as they arrive from the LLM.
    After all text is streamed, audio is yielded ONE sentence at a time
    in original sentence order — preventing parallel/overlapping voices.
    """
    tts_tasks: list[asyncio.Task[bytes]] = []  # ordered by sentence position
    buffer = ""

    # ── Phase 1: stream text tokens, launch TTS tasks as sentences complete ──
    async for token in text_stream:
        # Forward text token immediately (low time-to-first-token)
        yield token, None
        buffer += token

        complete_sentences, buffer = _split_complete_sentences(buffer)
        for sentence in complete_sentences:
            if sentence.strip():
                # Launch synthesis in background — do NOT await yet
                task: asyncio.Task[bytes] = asyncio.create_task(
                    synthesise_sentence(sentence.strip())
                )
                tts_tasks.append(task)
                logger.debug("TTS task started for sentence: %r", sentence[:40])

    # Flush any remaining text in buffer as the final sentence
    if buffer.strip():
        task = asyncio.create_task(synthesise_sentence(buffer.strip()))
        tts_tasks.append(task)
        logger.debug("TTS task started for final buffer: %r", buffer[:40])

    # ── Phase 2: yield audio IN ORDER, one sentence at a time ────────────────
    for i, task in enumerate(tts_tasks):
        try:
            audio_bytes = await task  # waits only if synthesis not done yet
            yield "", audio_bytes
            logger.debug("TTS audio yielded for sentence %d/%d", i + 1, len(tts_tasks))
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS failed for sentence %d: %s", i + 1, exc)
