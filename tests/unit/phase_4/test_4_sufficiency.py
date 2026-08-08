"""Tests SufficiencyChecker — méthode llm_local (mockée) + entity_matching (réel)."""

import pytest

from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage
from src.reliability.sufficiency import SufficiencyChecker

pytestmark = pytest.mark.phase4


class _FakeLLMClient(BaseLLMClient):
    def __init__(self, content: str):
        self.content = content

    def complete(self, messages, config):
        return LLMResponse(content=self.content, usage=LLMUsage(), model=config.model)

    async def complete_stream(self, messages, config):
        yield self.content

    def validate_config(self, config):
        return True


class TestSufficiencyCheckerLLM:
    def test_llm_verdict_is_used_when_client_available(self):
        client = _FakeLLMClient('{"verdict": "sufficient", "confidence": 0.9}')
        checker = SufficiencyChecker(llm_client=client)

        verdict = checker.check("Q?", [{"text": "context"}], LLMConfig(provider="ollama", model="x"))

        assert verdict.verdict == "sufficient"
        assert verdict.confidence == 0.9
        assert verdict.method == "llm_local"

    def test_falls_back_to_entity_matching_on_llm_failure(self):
        client = _FakeLLMClient("not json")
        checker = SufficiencyChecker(llm_client=client)

        verdict = checker.check(
            "What parameter is required?",
            [{"text": "The required parameter is symbol", "payload": {}}],
            LLMConfig(provider="ollama", model="x"),
        )

        assert verdict.method == "entity_matching"


class TestSufficiencyCheckerEntityMatching:
    def test_no_llm_client_uses_entity_matching(self):
        checker = SufficiencyChecker()
        verdict = checker.check(
            "What parameter is required for orders?",
            [{"text": "The required parameter for orders is symbol", "payload": {}}],
        )
        assert verdict.method == "entity_matching"
        assert verdict.verdict == "sufficient"

    def test_unrelated_context_is_insufficient(self):
        checker = SufficiencyChecker()
        verdict = checker.check(
            "What is the refund policy?",
            [{"text": "The weather today is sunny and warm", "payload": {}}],
        )
        assert verdict.verdict == "insufficient"

    def test_empty_context_is_insufficient(self):
        checker = SufficiencyChecker()
        verdict = checker.check("What parameter is required?", [])
        assert verdict.verdict == "insufficient"

    def test_partial_match_yields_partial_verdict(self):
        checker = SufficiencyChecker()
        verdict = checker.check(
            "What is the refund policy for cancelled subscriptions?",
            [{"text": "Subscriptions can be cancelled at any time", "payload": {}}],
        )
        assert verdict.verdict == "partial"
