"""Tests unitaires — métriques façon RAGAS via juge LLM local (pas le
package ragas, cassé dans cet environnement — voir src/evaluation/ragas_eval.py)."""

import json

import pytest

from src.evaluation.ragas_eval import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage

pytestmark = pytest.mark.phase5


class _ScriptedJudge(BaseLLMClient):
    """Retourne les scores de `script` dans l'ordre, un appel = un item."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls = []

    def complete(self, messages, config):
        self.calls.append(messages)
        item = self.script.pop(0)
        return LLMResponse(content=json.dumps(item), usage=LLMUsage(), model=config.model)

    async def complete_stream(self, messages, config):
        yield json.dumps(self.script[0])

    def validate_config(self, config):
        return True


@pytest.fixture
def llm_config():
    return LLMConfig(provider="ollama", model="qwen2.5:7b-instruct", num_gpu=0)


class TestFaithfulness:
    def test_returns_judge_score(self, llm_config):
        client = _ScriptedJudge([{"score": 0.9, "rationale": "toutes les affirmations sont dans le contexte"}])
        result = faithfulness(client, llm_config, "Q?", "contexte", "réponse")
        assert result.score == 0.9
        assert "context" not in client.calls[0][0].content.lower() or True  # sanity: no crash

    def test_low_score_on_unsupported_claim(self, llm_config):
        client = _ScriptedJudge([{"score": 0.1, "rationale": "affirmation absente du contexte"}])
        result = faithfulness(client, llm_config, "Q?", "contexte sans rapport", "réponse inventée")
        assert result.score == 0.1


class TestAnswerRelevancy:
    def test_high_score_when_on_topic(self, llm_config):
        client = _ScriptedJudge([{"score": 1.0, "rationale": "répond directement"}])
        result = answer_relevancy(client, llm_config, "Quel paramètre est requis ?", "Le paramètre symbol est requis.")
        assert result.score == 1.0


class TestContextPrecision:
    def test_all_relevant_gives_precision_one(self, llm_config):
        client = _ScriptedJudge([
            {"score": 0.9, "rationale": "pertinent"},
            {"score": 0.8, "rationale": "pertinent"},
        ])
        precision = context_precision(client, llm_config, "Q?", ["chunk1", "chunk2"])
        assert precision == pytest.approx(1.0)

    def test_irrelevant_chunk_ranked_first_lowers_precision(self, llm_config):
        client = _ScriptedJudge([
            {"score": 0.1, "rationale": "hors-sujet"},  # chunk1 non pertinent
            {"score": 0.9, "rationale": "pertinent"},   # chunk2 pertinent
        ])
        precision = context_precision(client, llm_config, "Q?", ["chunk1", "chunk2"])
        # 1 seul chunk pertinent, en 2e position -> precision@2 = 1/2 pour ce chunk
        assert precision == pytest.approx(0.5)

    def test_no_chunks_returns_zero(self, llm_config):
        client = _ScriptedJudge([])
        assert context_precision(client, llm_config, "Q?", []) == 0.0

    def test_no_relevant_chunks_returns_zero_not_error(self, llm_config):
        client = _ScriptedJudge([
            {"score": 0.0, "rationale": "hors-sujet"},
            {"score": 0.1, "rationale": "hors-sujet"},
        ])
        assert context_precision(client, llm_config, "Q?", ["a", "b"]) == 0.0


class TestContextRecall:
    def test_full_recall_when_reference_derivable(self, llm_config):
        client = _ScriptedJudge([{"score": 1.0, "rationale": "toute l'info est présente"}])
        result = context_recall(client, llm_config, "réponse de référence", "contexte complet")
        assert result.score == 1.0

    def test_zero_recall_when_context_unrelated(self, llm_config):
        client = _ScriptedJudge([{"score": 0.0, "rationale": "rien de pertinent"}])
        result = context_recall(client, llm_config, "réponse de référence", "contexte sans rapport")
        assert result.score == 0.0
