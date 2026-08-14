"""Tests unitaires — métriques façon RAGAS via juge LLM local (pas le
package ragas, cassé dans cet environnement — voir src/evaluation/ragas_eval.py).

Chaque métrique réplique l'algorithme réel de ragas (multi-étapes), donc les
scripts de juge simulent plusieurs appels LLM successifs par métrique, pas un
seul jugement holistique."""

import json

import numpy as np
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
    """Retourne les réponses de `script` dans l'ordre, un appel = un item."""

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


class _FakeBatch:
    def __init__(self, dense):
        self.dense = dense


class _FakeEmbedder:
    """Vecteurs déterministes pour tester la similarité cosinus sans charger BGE-M3."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed(self, texts):
        return _FakeBatch(np.array([self.vectors[t] for t in texts], dtype=float))


@pytest.fixture
def llm_config():
    return LLMConfig(provider="ollama", model="qwen2.5:7b-instruct", num_gpu=0)


class TestFaithfulness:
    def test_all_statements_supported(self, llm_config):
        client = _ScriptedJudge([
            {"statements": ["Le paramètre symbol est requis."]},
            {"statements": [{"statement": "Le paramètre symbol est requis.", "reason": "présent dans le contexte", "verdict": 1}]},
        ])
        result = faithfulness(client, llm_config, "Q?", "contexte mentionnant symbol", "réponse")
        assert result.score == 1.0
        assert len(client.calls) == 2

    def test_low_score_on_unsupported_claim(self, llm_config):
        client = _ScriptedJudge([
            {"statements": ["Affirmation inventée."]},
            {"statements": [{"statement": "Affirmation inventée.", "reason": "absente du contexte", "verdict": 0}]},
        ])
        result = faithfulness(client, llm_config, "Q?", "contexte sans rapport", "réponse inventée")
        assert result.score == 0.0

    def test_partial_support_averages_claims(self, llm_config):
        client = _ScriptedJudge([
            {"statements": ["Claim vraie.", "Claim fausse."]},
            {"statements": [
                {"statement": "Claim vraie.", "reason": "présente", "verdict": 1},
                {"statement": "Claim fausse.", "reason": "absente", "verdict": 0},
            ]},
        ])
        result = faithfulness(client, llm_config, "Q?", "contexte", "réponse")
        assert result.score == pytest.approx(0.5)

    def test_no_statements_returns_zero_without_second_call(self, llm_config):
        client = _ScriptedJudge([{"statements": []}])
        result = faithfulness(client, llm_config, "Q?", "contexte", "")
        assert result.score == 0.0
        assert len(client.calls) == 1


class TestAnswerRelevancy:
    def test_high_score_when_questions_align_with_original(self, llm_config):
        client = _ScriptedJudge([
            {"questions": [
                {"question": "Q1", "noncommittal": 0},
                {"question": "Q2", "noncommittal": 0},
                {"question": "Q3", "noncommittal": 0},
            ]}
        ])
        embedder = _FakeEmbedder({
            "Quel paramètre est requis ?": [1.0, 0.0],
            "Q1": [1.0, 0.0],
            "Q2": [1.0, 0.0],
            "Q3": [1.0, 0.0],
        })
        result = answer_relevancy(client, llm_config, "Quel paramètre est requis ?", "Le paramètre symbol est requis.", embedder)
        assert result.score == pytest.approx(1.0)

    def test_orthogonal_questions_give_low_score(self, llm_config):
        client = _ScriptedJudge([
            {"questions": [{"question": "Q1", "noncommittal": 0}]}
        ])
        embedder = _FakeEmbedder({
            "Question originale": [1.0, 0.0],
            "Q1": [0.0, 1.0],
        })
        result = answer_relevancy(client, llm_config, "Question originale", "réponse hors-sujet", embedder)
        assert result.score == pytest.approx(0.0, abs=1e-6)

    def test_noncommittal_answer_forces_zero_despite_similarity(self, llm_config):
        client = _ScriptedJudge([
            {"questions": [{"question": "Q1", "noncommittal": 1}]}
        ])
        embedder = _FakeEmbedder({
            "Question originale": [1.0, 0.0],
            "Q1": [1.0, 0.0],  # similarité parfaite, mais réponse évasive
        })
        result = answer_relevancy(client, llm_config, "Question originale", "Information non trouvée.", embedder)
        assert result.score == 0.0


class TestContextPrecision:
    def test_all_relevant_gives_precision_one(self, llm_config):
        client = _ScriptedJudge([
            {"reason": "utile", "verdict": 1},
            {"reason": "utile", "verdict": 1},
        ])
        precision = context_precision(client, llm_config, "Q?", ["chunk1", "chunk2"], "réponse")
        assert precision == pytest.approx(1.0)

    def test_irrelevant_chunk_ranked_first_lowers_precision(self, llm_config):
        client = _ScriptedJudge([
            {"reason": "hors-sujet", "verdict": 0},  # chunk1 non pertinent
            {"reason": "utile", "verdict": 1},        # chunk2 pertinent
        ])
        precision = context_precision(client, llm_config, "Q?", ["chunk1", "chunk2"], "réponse")
        # 1 seul chunk pertinent, en 2e position -> precision@2 = 1/2 pour ce chunk
        assert precision == pytest.approx(0.5)

    def test_no_chunks_returns_zero(self, llm_config):
        client = _ScriptedJudge([])
        assert context_precision(client, llm_config, "Q?", [], "réponse") == 0.0

    def test_no_relevant_chunks_returns_zero_not_error(self, llm_config):
        client = _ScriptedJudge([
            {"reason": "hors-sujet", "verdict": 0},
            {"reason": "hors-sujet", "verdict": 0},
        ])
        assert context_precision(client, llm_config, "Q?", ["a", "b"], "réponse") == 0.0


class TestContextRecall:
    def test_full_recall_when_reference_derivable(self, llm_config):
        client = _ScriptedJudge([
            {"classifications": [{"statement": "réponse de référence", "reason": "présente dans le contexte", "attributed": 1}]}
        ])
        result = context_recall(client, llm_config, "réponse de référence", "contexte complet")
        assert result.score == 1.0

    def test_zero_recall_when_context_unrelated(self, llm_config):
        client = _ScriptedJudge([
            {"classifications": [{"statement": "réponse de référence", "reason": "absente du contexte", "attributed": 0}]}
        ])
        result = context_recall(client, llm_config, "réponse de référence", "contexte sans rapport")
        assert result.score == 0.0

    def test_partial_recall_averages_sentences(self, llm_config):
        client = _ScriptedJudge([
            {"classifications": [
                {"statement": "phrase couverte", "reason": "présente", "attributed": 1},
                {"statement": "phrase non couverte", "reason": "absente", "attributed": 0},
            ]}
        ])
        result = context_recall(client, llm_config, "phrase couverte. phrase non couverte.", "contexte partiel")
        assert result.score == pytest.approx(0.5)
