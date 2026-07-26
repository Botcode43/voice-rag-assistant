"""
app/dependencies.py — FastAPI dependency injection helpers.
"""

from __future__ import annotations

from app.config import Settings, get_settings


def get_app_settings() -> Settings:
    """FastAPI dependency that returns the application settings."""
    return get_settings()
