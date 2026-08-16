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


class TestSufficiencyCheckerDiffExplanation:
    """Angle mort corrigé : sans diff_explanation, une question de conflit
    de version ("qu'est-ce qui a changé entre X et Y") est jugée
    insufficient car les chunks bruts (une seule version) ne peuvent
    effectivement pas y répondre seuls — même si le diff précalculé, lui,
    le peut. Le diff doit maintenant être factorisé dans le jugement."""

    def test_entity_matching_becomes_sufficient_when_diff_covers_the_gap(self):
        checker = SufficiencyChecker()
        question = "Qu'est-ce qui a changé pour le paramètre refundamount entre les versions ?"
        context = [{"text": "Endpoint description sans rapport", "payload": {}}]

        without_diff = checker.check(question, context)
        with_diff = checker.check(
            question, context,
            diff_explanation="Changements entre legacy et v2213:\n- refundamount: modifié (description mise à jour)",
        )

        assert without_diff.verdict == "insufficient"
        assert with_diff.verdict in ("sufficient", "partial")

    def test_llm_path_includes_diff_in_judged_context(self):
        captured = {}

        class _CapturingLLM(BaseLLMClient):
            def complete(self, messages, config):
                captured["user_content"] = messages[-1].content
                return LLMResponse(content='{"verdict": "sufficient", "confidence": 0.9}', usage=LLMUsage(), model=config.model)

            async def complete_stream(self, messages, config):
                yield ""

            def validate_config(self, config):
                return True

        checker = SufficiencyChecker(llm_client=_CapturingLLM())
        checker.check(
            "Q?", [{"text": "context"}], LLMConfig(provider="ollama", model="x"),
            diff_explanation="Changements entre legacy et v2213:\n- foo: modifié",
        )
        assert "Changements entre legacy et v2213" in captured["user_content"]

    def test_no_diff_explanation_behaves_exactly_as_before(self):
        """Rétrocompatibilité stricte : diff_explanation=None (défaut) ne
        change rien au comportement existant."""
        checker = SufficiencyChecker()
        verdict = checker.check(
            "What parameter is required for orders?",
            [{"text": "The required parameter for orders is symbol", "payload": {}}],
        )
        assert verdict.verdict == "sufficient"
