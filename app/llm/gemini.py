"""
app/llm/gemini.py — Gemini 2.5 Flash streaming generation.

Uses google-genai SDK (v1.x) with async streaming.
Yields text tokens as they arrive from the model.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.prompts import SYSTEM_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level Gemini client — created once per process
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client  # noqa: PLW0603
    if _client is None:
        settings = get_settings()
        _client = genai.Client(api_key=settings.google_api_key)
        logger.info("Gemini client initialised (model=%s)", settings.gemini_model)
    return _client


async def stream_generate_v2(prompt: str) -> AsyncIterator[str]:
    """
    Async streaming from gemini-flash-latest.

    Uses the async client (client.aio) for proper async support.
    Falls back to thread-based streaming if async is unavailable.

    Yields text token strings as they arrive.
    """
    settings = get_settings()
    client = _get_client()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=settings.gemini_temperature,
    )

    logger.debug("Starting Gemini stream (prompt_len=%d)", len(prompt))

    try:
        # google-genai 1.x — async streaming via client.aio
        async for chunk in client.aio.models.generate_content_stream(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
    except AttributeError:
        # Fallback for SDK versions without aio namespace
        logger.warning("client.aio not available — using thread-based streaming fallback")
        async for token in _thread_stream(client, settings.gemini_model, prompt, config):
            yield token


async def _thread_stream(
    client: genai.Client,
    model: str,
    prompt: str,
    config: types.GenerateContentConfig,
) -> AsyncIterator[str]:
    """Run sync streaming in a thread, bridge to async via a queue."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _producer() -> None:
        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=config,
            ):
                if chunk.text:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as exc:
            logger.exception("Gemini producer error: %s", exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(asyncio.to_thread(_producer))

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token
