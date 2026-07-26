"""
app/guardrails/prompt_guard.py — Fast regex-based prompt injection detection.

No LLM call is made here.  If any pattern matches, the request is immediately
short-circuited before entering the RAG pipeline.
"""

from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Injection patterns (case-insensitive) ──────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|prior|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output|display)\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"show\s+(me\s+)?your\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(the\s+)?(uploaded\s+)?document", re.IGNORECASE),
    re.compile(r"ignore\s+(the\s+)?(pdf|document|uploaded)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(chatgpt|gpt|openai|claude|llm)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be|you're)", re.IGNORECASE),
    re.compile(r"tell\s+me\s+(your\s+)?(hidden|secret|real)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?different", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|context)", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(instructions?|settings?|rules?)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"bypass\s+(your\s+)?(safety|guard|filter|restriction)", re.IGNORECASE),
]

INJECTION_REJECTION_MESSAGE = (
    "Your request was blocked because it appears to be a prompt injection attempt. "
    "Please ask a question about the uploaded document."
)


def is_injection(query: str) -> bool:
    """Return True if *query* matches any known injection pattern."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Prompt injection detected: pattern=%s query=%r", pattern.pattern, query[:80])
            return True
    return False
