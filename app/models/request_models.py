"""
app/models/request_models.py — Pydantic request schemas for FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    session_id: str = Field(
        ..., description="Unique session identifier for memory scoping"
    )
    pdf_hash: str = Field(
        ..., description="SHA-256 hash of the uploaded PDF (returned by /upload)"
    )
    query: str = Field(..., min_length=1, description="User question text")


class VoiceQueryRequest(BaseModel):
    """Payload for POST /voice-query — audio bytes come as form data."""

    session_id: str = Field(..., description="Unique session identifier")
    pdf_hash: str = Field(..., description="SHA-256 hash of the uploaded PDF")
