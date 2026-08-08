"""Tests LLMFallbackDetector — vote majoritaire, auto-cohérence."""

import json
from uuid import uuid4

import pytest

from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage
from src.reliability.conflict.llm_fallback import LLMFallbackDetector

pytestmark = pytest.mark.phase4


class _ScriptedLLMClient(BaseLLMClient):
    """Retourne les réponses de `script` dans l'ordre, un appel = un item."""

    def __init__(self, script: list[dict]):
        self.script = script
        self.call_count = 0
        self.temperatures_seen: list[float] = []

    def complete(self, messages, config):
        self.temperatures_seen.append(config.temperature)
        payload = self.script[self.call_count]
        self.call_count += 1
        if isinstance(payload, Exception):
            raise payload
        return LLMResponse(content=json.dumps(payload), usage=LLMUsage(), model=config.model)

    async def complete_stream(self, messages, config):
        yield ""

    def validate_config(self, config):
        return True


def _chunk(source_id: str, text: str = "text"):
    return {"chunk_id": str(uuid4()), "text": text, "payload": {"source_id": source_id}}


class TestLLMFallbackDetector:
    def test_majority_conflict_wins(self):
        script = [
            {"conflict": True, "type": "factual", "explanation": "they disagree"},
            {"conflict": True, "type": "factual", "explanation": "yes conflict"},
            {"conflict": False, "type": None, "explanation": None},
        ]
        client = _ScriptedLLMClient(script)
        detector = LLMFallbackDetector(client, n_samples=3)

        report = detector.detect(_chunk("a"), _chunk("b"), LLMConfig(provider="ollama", model="x"))

        assert report.conflict is True
        assert report.method == "llm_fallback"
        assert report.type == "factual"
        assert report.confidence == pytest.approx(2 / 3, abs=0.01)
        assert report.explanation is not None
        assert len(report.chunks) == 2

    def test_majority_no_conflict_wins(self):
        script = [
            {"conflict": False, "type": None, "explanation": None},
            {"conflict": False, "type": None, "explanation": None},
            {"conflict": True, "type": "factual", "explanation": "outlier"},
        ]
        client = _ScriptedLLMClient(script)
        detector = LLMFallbackDetector(client, n_samples=3)

        report = detector.detect(_chunk("a"), _chunk("b"), LLMConfig(provider="ollama", model="x"))

        assert report.conflict is False
        assert report.type is None

    def test_samples_use_varied_temperatures(self):
        script = [{"conflict": False, "type": None, "explanation": None}] * 3
        client = _ScriptedLLMClient(script)
        detector = LLMFallbackDetector(client, n_samples=3)

        detector.detect(_chunk("a"), _chunk("b"), LLMConfig(provider="ollama", model="x"))

        assert len(set(client.temperatures_seen)) > 1

    def test_failed_samples_are_skipped_not_fatal(self):
        script = [
            RuntimeError("timeout"),
            {"conflict": True, "type": "temporal", "explanation": "differs by version"},
            {"conflict": True, "type": "temporal", "explanation": "differs by version"},
        ]
        client = _ScriptedLLMClient(script)
        detector = LLMFallbackDetector(client, n_samples=3)

        report = detector.detect(_chunk("a"), _chunk("b"), LLMConfig(provider="ollama", model="x"))

        assert report.conflict is True
        assert report.type == "temporal"

    def test_all_samples_failing_returns_no_conflict(self):
        script = [RuntimeError("down")] * 3
        client = _ScriptedLLMClient(script)
        detector = LLMFallbackDetector(client, n_samples=3)

        report = detector.detect(_chunk("a"), _chunk("b"), LLMConfig(provider="ollama", model="x"))

        assert report.conflict is False
        assert report.method == "llm_fallback"
