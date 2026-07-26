"""
app/main.py — FastAPI application entry point.

Registers routers, startup/shutdown lifecycle hooks, and global exception handlers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import query as query_router
from app.api import upload as upload_router
from app.api import voice as voice_router
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    settings = get_settings()

    # Ensure required directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    if settings.qdrant_path != ":memory:":
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)

    logger.info("Voice RAG Assistant starting up …")
    logger.info(
        "Config: model=%s embedding=%s qdrant_path=%s",
        settings.gemini_model,
        settings.embedding_model,
        settings.qdrant_path,
    )

    # ── Warm-up: repopulate in-memory pdf_cache from existing Qdrant collections ──
    try:
        from app.rag.vector_store import get_qdrant_client
        from app.cache.cache import pdf_cache

        client = get_qdrant_client()
        existing = await client.get_collections()
        for coll in existing.collections:
            name = coll.name  # e.g. "rag_<16-char-hash>"
            if name.startswith("rag_"):
                reconstructed_hash = name[4:]  # strip "rag_" prefix
                if not pdf_cache.has(reconstructed_hash):
                    pdf_cache.set(reconstructed_hash, name)
                    logger.info("Warm-up: restored cache entry %s -> %s", reconstructed_hash, name)
        logger.info("Cache warm-up complete: %d collections found", len(existing.collections))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache warm-up failed (non-fatal): %s", exc)

    yield  # Application runs here

    logger.info("Voice RAG Assistant shutting down …")
    # Close the persistent Ollama HTTP client gracefully
    try:
        from app.rag.embeddings import close_http_client
        await close_http_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error closing HTTP client (non-fatal): %s", exc)


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Voice RAG Assistant",
        description=(
            "Upload a PDF and ask questions — answered with streamed text and voice, "
            "grounded ONLY in the document content."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS (allow Streamlit frontend) ────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(upload_router.router, tags=["PDF Upload"])
    app.include_router(query_router.router, tags=["Query"])
    app.include_router(voice_router.router, tags=["Voice Query"])

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    # ── Health check ───────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "model": settings.gemini_model}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
