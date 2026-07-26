"""
app/cache/cache.py — In-memory caches for PDF processing and LLM responses.

Two caches are implemented as module-level singletons:
  1. pdf_cache   — maps sha256(pdf_bytes) → collection_name
  2. llm_cache   — maps sha256(pdf_hash + norm_query) → full answer text
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe LRU cache backed by an OrderedDict."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: str, value: Any) -> None:  # noqa: A003
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                evicted_key, _ = self._store.popitem(last=False)
                logger.info("LRU evicted key=%s", evicted_key)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── Module-level singletons ────────────────────────────────────────────────────

# pdf_cache: pdf_hash → Qdrant collection name
pdf_cache: LRUCache = LRUCache(maxsize=64)

# llm_cache: cache_key → full answer string
llm_cache: LRUCache = LRUCache(maxsize=256)
