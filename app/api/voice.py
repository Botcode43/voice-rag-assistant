"""
app/api/voice.py — POST /voice-query endpoint.

Accepts uploaded audio bytes, transcribes them with SpeechRecognition,
then runs the same pipeline as POST /query.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.services.rag_service import handle_query
from app.speech.speech_to_text import transcribe_audio
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/voice-query",
    summary="Ask a voice question about the uploaded PDF",
    response_class=StreamingResponse,
)
async def voice_query_endpoint(
    audio: UploadFile = File(..., description="Audio file (WAV preferred)"),
    pdf_hash: str = Form(..., description="SHA-256 hash from POST /upload"),
    session_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier",
    ),
) -> StreamingResponse:
    """
    Submit a voice query:
    1. Receives an audio file upload.
    2. Transcribes via SpeechRecognition (Google STT).
    3. Runs the same RAG pipeline as POST /query.
    4. Returns an SSE stream of text + audio events.
    """
    # Read audio bytes
    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {exc}",
        ) from exc

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty.",
        )

    # Transcribe
    content_type = audio.content_type or "audio/wav"
    try:
        query_text = await transcribe_audio(audio_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("STT service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    logger.info(
        "Voice query transcribed: session=%s text=%r", session_id, query_text[:80]
    )

    # Run RAG pipeline
    try:
        event_stream = await handle_query(
            session_id=session_id,
            pdf_hash=pdf_hash,
            query=query_text,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /voice-query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from exc

    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Transcribed-Query": query_text[:200],  # for debugging
        },
    )
