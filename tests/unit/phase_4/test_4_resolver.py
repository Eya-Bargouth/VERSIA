"""Tests resolver.py — stratégie de résolution selon le type de conflit."""

import pytest

from src.reliability.conflict.resolver import resolve
from src.reliability.conflict.types import ConflictReport

pytestmark = pytest.mark.phase4


def _report(**kwargs):
    defaults = dict(conflict=True, method="llm_fallback")
    defaults.update(kwargs)
    return ConflictReport(**defaults)


class TestResolve:
    def test_no_conflict_returns_none_strategy(self):
        resolution = resolve(_report(conflict=False), {"payload": {}}, {"payload": {}})
        assert resolution.strategy == "none"

    def test_factual_prefers_more_recent_by_valid_until(self):
        chunk_a = {"chunk_id": "a", "payload": {"valid_until": "2025-01-01"}}
        chunk_b = {"chunk_id": "b", "payload": {"valid_until": "2026-01-01"}}
        resolution = resolve(_report(type="factual"), chunk_a, chunk_b)
        assert resolution.strategy == "prefer_recent"
        assert resolution.preferred_chunk_id == "b"

    def test_temporal_prefers_more_recent_by_version_order(self):
        chunk_a = {"chunk_id": "a", "payload": {"version_order": 0}}
        chunk_b = {"chunk_id": "b", "payload": {"version_order": 2}}
        resolution = resolve(_report(type="temporal"), chunk_a, chunk_b)
        assert resolution.strategy == "prefer_recent"
        assert resolution.preferred_chunk_id == "b"

    def test_factual_undetermined_when_no_signal(self):
        chunk_a = {"chunk_id": "a", "payload": {}}
        chunk_b = {"chunk_id": "b", "payload": {}}
        resolution = resolve(_report(type="factual"), chunk_a, chunk_b)
        assert resolution.strategy == "undetermined"

    def test_opinion_uses_multi_perspective(self):
        resolution = resolve(_report(type="opinion"), {"payload": {}}, {"payload": {}})
        assert resolution.strategy == "multi_perspective"
