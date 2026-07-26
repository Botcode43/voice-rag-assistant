"""
tests/test_helpers.py — Unit tests for utility helpers.
"""
import pytest
from app.utils.helpers import normalize_query, make_llm_cache_key, split_into_sentences, sha256_bytes


def test_sha256_bytes():
    h = sha256_bytes(b"hello")
    assert len(h) == 64
    assert h == sha256_bytes(b"hello")
    assert h != sha256_bytes(b"world")


def test_normalize_query_lowercase():
    assert normalize_query("HELLO WORLD") == "hello world"


def test_normalize_query_strips_punctuation():
    result = normalize_query("Who wrote this?!")
    assert "?" not in result
    assert "!" not in result


def test_normalize_query_collapses_whitespace():
    assert normalize_query("  hello   world  ") == "hello world"


def test_make_llm_cache_key_stable():
    key1 = make_llm_cache_key("abc123", "What is the main topic?")
    key2 = make_llm_cache_key("abc123", "what is the main topic")
    assert key1 == key2  # normalized queries should produce same key


def test_make_llm_cache_key_different_pdfs():
    key1 = make_llm_cache_key("hash_a", "same question")
    key2 = make_llm_cache_key("hash_b", "same question")
    assert key1 != key2


def test_split_sentences():
    text = "This is sentence one. This is sentence two! Is this three?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_single():
    assert split_into_sentences("Hello world.") == ["Hello world."]
