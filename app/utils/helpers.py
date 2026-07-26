"""
app/utils/helpers.py — Shared utilities used across modules.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata


def sha256_bytes(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    """Return the hex-encoded SHA-256 digest of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def normalize_query(text: str) -> str:
    """
    Lowercase, strip extra whitespace, remove diacritics, collapse punctuation.
    Used as the normalisation step before building the LLM cache key.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_llm_cache_key(pdf_hash: str, query: str) -> str:
    """Build a stable cache key for an (pdf_hash, query) pair."""
    normalized = normalize_query(query)
    raw = f"{pdf_hash}::{normalized}"
    return sha256_str(raw)


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that handles common abbreviations.
    Returns a list of sentence strings (stripped, non-empty).
    """
    # Split on sentence-terminating punctuation followed by whitespace or end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
