"""Tests ConflictDetector — orchestration structurel + textuel."""

from uuid import uuid4

import pytest

from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage
from src.reliability.conflict.detector import ConflictDetector, _cross_source_pairs

pytestmark = pytest.mark.phase4


def _chunk(source_id: str):
    return {"chunk_id": str(uuid4()), "text": "t", "payload": {"source_id": source_id}}


class _AlwaysConflictClient(BaseLLMClient):
    def complete(self, messages, config):
        import json

        return LLMResponse(
            content=json.dumps({"conflict": True, "type": "factual", "explanation": "x"}),
            usage=LLMUsage(),
            model=config.model,
        )

    async def complete_stream(self, messages, config):
        yield ""

    def validate_config(self, config):
        return True


class TestCrossSourcePairs:
    def test_only_pairs_different_sources(self):
        chunks = [_chunk("a"), _chunk("a"), _chunk("b")]
        pairs = list(_cross_source_pairs(chunks))
        assert len(pairs) == 2  # (a1,b), (a2,b) — jamais (a1,a2)
        for ca, cb in pairs:
            assert ca["payload"]["source_id"] != cb["payload"]["source_id"]

    def test_no_pairs_for_single_source(self):
        chunks = [_chunk("a"), _chunk("a")]
        assert list(_cross_source_pairs(chunks)) == []


class TestConflictDetectorTextual:
    def test_no_llm_client_returns_empty(self):
        detector = ConflictDetector(llm_client=None)
        result = detector.detect_textual([_chunk("a"), _chunk("b")], LLMConfig(provider="ollama", model="x"))
        assert result == []

    def test_only_conflicting_pairs_are_returned(self):
        detector = ConflictDetector(llm_client=_AlwaysConflictClient())
        chunks = [_chunk("a"), _chunk("b"), _chunk("a")]
        result = detector.detect_textual(chunks, LLMConfig(provider="ollama", model="x"))
        assert len(result) == 2
        assert all(r.conflict for r in result)


class TestConflictDetectorStructural:
    def test_delegates_to_version_diff_detector(self, tmp_path):
        detector = ConflictDetector(diff_dir=tmp_path)
        report = detector.detect_structural("unknown_source", "v1", "v2")
        assert report.conflict is False
        assert report.method == "version_diff"
