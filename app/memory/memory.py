"""
app/memory/memory.py — In-memory session conversation store.

Each session_id maps to a list of turn dicts:
    [{"role": "user"|"assistant", "content": str}, ...]

Access is thread-safe. Memory is never persisted to disk.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionMemory:
    """Per-session turn-based conversation memory."""

    def __init__(self, max_turns: int = 10) -> None:
        self._max_turns = max_turns
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a turn; trim to max_turns (pairs, so 2*max_turns messages)."""
        with self._lock:
            turns = self._store[session_id]
            turns.append({"role": role, "content": content})
            # Keep only the last max_turns * 2 messages (user + assistant pairs)
            if len(turns) > self._max_turns * 2:
                self._store[session_id] = turns[-(self._max_turns * 2):]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the turn history for a session (read-only copy)."""
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Clear the history for a session."""
        with self._lock:
            self._store.pop(session_id, None)

    def format_for_context(self, session_id: str) -> str:
        """Return a formatted string representation suitable for LLM context."""
        history = self.get_history(session_id)
        if not history:
            return ""
        lines: list[str] = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)


# Module-level singleton — one memory store for the entire process
session_memory: SessionMemory = SessionMemory()
