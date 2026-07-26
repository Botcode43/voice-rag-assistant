"""
tests/test_cache.py — Unit tests for the LRU cache.
"""
import pytest
from app.cache.cache import LRUCache


def test_set_and_get():
    cache = LRUCache(maxsize=3)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"


def test_miss_returns_none():
    cache = LRUCache(maxsize=3)
    assert cache.get("nonexistent") is None


def test_has():
    cache = LRUCache(maxsize=3)
    cache.set("key", "value")
    assert cache.has("key") is True
    assert cache.has("missing") is False


def test_lru_eviction():
    cache = LRUCache(maxsize=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")  # k1 should be evicted
    assert cache.has("k1") is False
    assert cache.has("k2") is True
    assert cache.has("k3") is True


def test_access_refreshes_lru():
    cache = LRUCache(maxsize=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    _ = cache.get("k1")   # refresh k1
    cache.set("k3", "v3")  # k2 should be evicted now, not k1
    assert cache.has("k1") is True
    assert cache.has("k2") is False


def test_len():
    cache = LRUCache(maxsize=5)
    assert len(cache) == 0
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2
