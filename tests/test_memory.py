"""
tests/test_memory.py — Unit tests for session memory.
"""
import pytest
from app.memory.memory import SessionMemory


def test_add_and_get_history():
    mem = SessionMemory(max_turns=5)
    mem.add_turn("sess1", "user", "Hello")
    mem.add_turn("sess1", "assistant", "Hi there!")
    history = mem.get_history("sess1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_max_turns_trimming():
    mem = SessionMemory(max_turns=2)
    for i in range(5):
        mem.add_turn("s", "user", f"Q{i}")
        mem.add_turn("s", "assistant", f"A{i}")
    history = mem.get_history("s")
    # Max 2 turns = 4 messages
    assert len(history) == 4


def test_clear():
    mem = SessionMemory()
    mem.add_turn("sess", "user", "x")
    mem.clear("sess")
    assert mem.get_history("sess") == []


def test_format_for_context_empty():
    mem = SessionMemory()
    assert mem.format_for_context("new_session") == ""


def test_format_for_context():
    mem = SessionMemory()
    mem.add_turn("s", "user", "Question")
    mem.add_turn("s", "assistant", "Answer")
    ctx = mem.format_for_context("s")
    assert "User: Question" in ctx
    assert "Assistant: Answer" in ctx


def test_independent_sessions():
    mem = SessionMemory()
    mem.add_turn("sess_a", "user", "A question")
    mem.add_turn("sess_b", "user", "B question")
    assert len(mem.get_history("sess_a")) == 1
    assert len(mem.get_history("sess_b")) == 1
