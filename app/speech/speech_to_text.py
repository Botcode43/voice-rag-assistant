"""
app/speech/speech_to_text.py — Speech-to-text using the SpeechRecognition library.

Audio bytes (WAV/WebM/MP3) are converted to text using Google's free STT API.
This runs synchronously in a thread pool since SpeechRecognition is blocking.
"""

from __future__ import annotations

import io
import asyncio
import tempfile
import os

import speech_recognition as sr

from app.utils.logger import get_logger

logger = get_logger(__name__)


async def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    """
    Transcribe audio bytes to text using SpeechRecognition (Google STT).

    Args:
        audio_bytes: Raw audio file bytes.
        content_type: MIME type of the audio (e.g. 'audio/wav', 'audio/webm').

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If audio cannot be transcribed.
        RuntimeError: If STT service is unavailable.
    """
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, content_type)


def _transcribe_sync(audio_bytes: bytes, content_type: str) -> str:
    """Blocking transcription — intended to be run in a thread pool."""
    recogniser = sr.Recognizer()

    # Write to a temp file since SpeechRecognition reads from file paths / AudioFile
    suffix = _suffix_for(content_type)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            recogniser.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recogniser.record(source)

        text = recogniser.recognize_google(audio_data)
        logger.info("STT transcribed: %r", text[:100])
        return text
    except sr.UnknownValueError as exc:
        raise ValueError("Could not understand the audio. Please speak more clearly.") from exc
    except sr.RequestError as exc:
        raise RuntimeError(f"Google STT service error: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _suffix_for(content_type: str) -> str:
    """Map a MIME type to a file extension."""
    mapping = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
    }
    return mapping.get(content_type.split(";")[0].strip().lower(), ".wav")
