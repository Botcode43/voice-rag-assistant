"""
app/config.py — Application configuration loaded from environment variables.
All settings are Pydantic-validated and read from .env at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration — load from .env, then environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash", description="Gemini model identifier"
    )
    gemini_temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Set EMBEDDING_PROVIDER=google to use Google's text-embedding-004 (fast, same API key).
    # Set EMBEDDING_PROVIDER=ollama to use local Ollama nomic-embed-text (requires Ollama running).
    embedding_provider: str = Field(
        default="ollama",
        description="Embedding backend: 'ollama' (local) or 'google' (Google Generative AI)",
    )
    # ── Ollama settings (used when embedding_provider=ollama) ─────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )
    embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama embedding model name"
    )
    # ── Google embedding settings (used when embedding_provider=google) ───────
    google_embedding_model: str = Field(
        default="text-embedding-004",
        description="Google Generative AI embedding model (e.g. text-embedding-004)",
    )
    embedding_dim: int = Field(default=768, description="Embedding vector dimension")

    # ── Vector DB ─────────────────────────────────────────────────────────────
    qdrant_collection: str = Field(
        default="voice_rag", description="Qdrant collection name prefix"
    )
    qdrant_path: str = Field(
        default="./qdrant_storage",
        description="Path for Qdrant local persistent storage (use ':memory:' for pure in-memory)",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=5, description="Number of chunks to retrieve per query"
    )
    grounding_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum RRF score from Qdrant hybrid search to consider retrieval grounded. "
            "RRF scores are typically in the 0.01-0.05 range — keep this low."
        ),
    )
    dense_similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Score threshold for dense Prefetch before RRF fusion. "
            "Set to 0.0 to disable pre-filtering and let RRF pick the best candidates."
        ),
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=700, description="Token-aware chunk size")
    chunk_overlap: int = Field(default=120, description="Token-aware chunk overlap")

    # ── Paths ─────────────────────────────────────────────────────────────────
    upload_dir: Path = Field(
        default=Path("./uploads"), description="Directory for uploaded PDFs"
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_voice: str = Field(
        default="en-US-AriaNeural",
        description="Edge-TTS voice identifier",
    )

    # ── Session memory ────────────────────────────────────────────────────────
    memory_max_turns: int = Field(
        default=10, description="Max conversation turns kept in session memory"
    )

    # ── LLM response cache ────────────────────────────────────────────────────
    llm_cache_max_size: int = Field(
        default=256, description="Max number of cached LLM responses (in-memory)"
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
