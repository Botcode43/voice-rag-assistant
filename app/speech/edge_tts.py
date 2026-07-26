"""
app/speech/edge_tts.py — Per-sentence TTS streaming via edge-tts.

Audio is generated as MP3 bytes for each completed sentence and yielded
immediately — we do NOT wait for the full LLM answer.
"""

from __future__ import annotations

import asyncio
import io

import edge_tts

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def synthesise_sentence(text: str) -> bytes:
    """
    Synthesise a single sentence to MP3 bytes using edge-tts.

    Args:
        text: The sentence text to synthesise.

    Returns:
        MP3 audio bytes.

    Raises:
        RuntimeError: If edge-tts synthesis fails.
    """
    settings = get_settings()
    voice = settings.tts_voice

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio_bytes = buf.getvalue()
        logger.debug(
            "TTS synthesised sentence (len=%d chars) → %d audio bytes",
            len(text),
            len(audio_bytes),
        )
        return audio_bytes
    except Exception as exc:
        logger.exception("edge-tts synthesis failed for text=%r: %s", text[:40], exc)
        raise RuntimeError(f"TTS synthesis failed: {exc}") from exc


async def synthesise_text(text: str) -> bytes:
    """Synthesise a full text (no sentence splitting — use for short strings)."""
    return await synthesise_sentence(text)
