"""
app/utils/logger.py — Structured logging factory.
All application code should import get_logger() rather than using logging directly.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache


class _JsonFormatter(logging.Formatter):
    """Minimal JSON-like formatter that avoids external deps."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        message = super().format(record)
        return (
            f'{{"time":"{self.formatTime(record)}","level":"{record.levelname}",'
            f'"name":"{record.name}","message":{message!r}}}'
        )


@lru_cache(maxsize=None)
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger configured with structured output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Force UTF-8 on Windows (default cp1252 chokes on Unicode log chars)
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass  # Not a reconfigurable stream — no-op
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
