"""
app/models/response_models.py — Pydantic response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response from POST /upload."""

    pdf_hash: str = Field(..., description="SHA-256 hash of the uploaded PDF")
    filename: str = Field(..., description="Original filename")
    cached: bool = Field(
        ...,
        description="True if this PDF was already processed and the cache was reused",
    )
    num_chunks: int = Field(
        ..., description="Number of text chunks stored in the vector DB"
    )
    message: str = Field(..., description="Human-readable status message")


class StreamChunk(BaseModel):
    """A single streamed SSE event payload (serialised as JSON in the data field)."""

    type: str = Field(
        ...,
        description="Event type: 'text' | 'audio' | 'done' | 'error'",
    )
    content: str | None = Field(None, description="Text content (for type='text')")
    audio_b64: str | None = Field(
        None, description="Base64-encoded MP3 audio (for type='audio')"
    )
    error: str | None = Field(None, description="Error message (for type='error')")
