"""
app/api/upload.py — POST /upload endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.response_models import UploadResponse
from app.services.pdf_service import process_pdf
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF and process it for RAG",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
) -> UploadResponse:
    """
    Upload a PDF file, parse it, chunk it, embed it, and store it in Qdrant.

    Returns the pdf_hash which must be passed with subsequent /query calls.
    If the same PDF has been uploaded before (identified by sha256 hash),
    re-processing is skipped and the cached result is returned immediately.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Allow octet-stream for clients that don't set content type properly
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are supported.",
            )

    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        pdf_hash, collection_name, was_cached, num_chunks = await process_pdf(
            pdf_bytes, file.filename or "unknown.pdf"
        )
    except ValueError as exc:
        logger.warning("PDF processing validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("PDF processing runtime error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    msg = (
        f"PDF loaded from cache ({num_chunks} chunks)."
        if was_cached
        else f"PDF processed successfully ({num_chunks} chunks embedded)."
    )

    return UploadResponse(
        pdf_hash=pdf_hash,
        filename=file.filename or "unknown.pdf",
        cached=was_cached,
        num_chunks=num_chunks,
        message=msg,
    )
