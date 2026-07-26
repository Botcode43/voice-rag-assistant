"""
app/api/query.py — POST /query and GET /stream endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.models.request_models import QueryRequest
from app.services.rag_service import handle_query
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    summary="Ask a text question about the uploaded PDF",
    response_class=StreamingResponse,
)
async def query_endpoint(request: QueryRequest) -> StreamingResponse:
    """
    Submit a text query against a previously uploaded PDF.

    Returns an SSE (Server-Sent Events) stream with JSON payloads:
    - ``{"type": "text", "content": "..."}`` — streamed answer tokens
    - ``{"type": "audio", "audio_b64": "..."}`` — base64 MP3 audio per sentence
    - ``{"type": "done"}`` — signals stream completion
    - ``{"type": "error", "error": "..."}`` — error event

    The ``pdf_hash`` must be the value returned by a prior POST /upload call.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty.",
        )

    try:
        event_stream = await handle_query(
            session_id=request.session_id,
            pdf_hash=request.pdf_hash,
            query=request.query,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /query: %s", exc)
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
        },
    )


@router.get(
    "/stream",
    summary="Generic streaming endpoint (health-check / demo)",
    response_class=StreamingResponse,
)
async def stream_endpoint(
    session_id: str = Query(default_factory=lambda: str(uuid.uuid4())),
    pdf_hash: str = Query(...),
    query: str = Query(...),
) -> StreamingResponse:
    """
    GET variant of the query endpoint — useful for EventSource-based clients
    where setting a request body is awkward.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty.",
        )

    try:
        event_stream = await handle_query(
            session_id=session_id,
            pdf_hash=pdf_hash,
            query=query,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /stream: %s", exc)
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
        },
    )
