"""Tests unitaires — Citation Accuracy et Sufficiency Precision (spec §15.3,
audit #20)."""

import json

import pytest

from src.evaluation.reliability_metrics import citation_accuracy, sufficiency_precision
from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage

pytestmark = pytest.mark.phase5


class _ScriptedJudge(BaseLLMClient):
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


class _Citation:
    def __init__(self, text_span: str):
        self.text_span = text_span


@pytest.fixture
def llm_config():
    return LLMConfig(provider="ollama", model="qwen2.5:7b-instruct", num_gpu=0)


class TestCitationAccuracy:
    def test_all_citations_supported(self, llm_config):
        client = _ScriptedJudge([
            {"reason": "soutient l'affirmation", "verdict": 1},
            {"reason": "soutient l'affirmation", "verdict": 1},
        ])
        citations = [_Citation("le paramètre symbol est requis"), _Citation("interval est un paramètre")]
        result = citation_accuracy(client, llm_config, "Réponse générée.", citations)
        assert result["citation_accuracy"] == 1.0
        assert result["n_citations"] == 2

    def test_partial_support(self, llm_config):
        client = _ScriptedJudge([
            {"reason": "soutient", "verdict": 1},
            {"reason": "hors-sujet", "verdict": 0},
        ])
        citations = [_Citation("a"), _Citation("b")]
        result = citation_accuracy(client, llm_config, "Réponse.", citations)
        assert result["citation_accuracy"] == 0.5
        assert result["n_accurate"] == 1

    def test_no_citations_returns_none_not_zero(self, llm_config):
        client = _ScriptedJudge([])
        result = citation_accuracy(client, llm_config, "Réponse.", [])
        assert result["citation_accuracy"] is None
        assert result["n_citations"] == 0
        assert client.calls == []

    def test_accepts_dict_citations_too(self, llm_config):
        """Compatible avec des citations sous forme de dict (pas seulement
        des objets Citation), pour rester utilisable depuis des données déjà
        sérialisées (JSONL de baseline)."""
        client = _ScriptedJudge([{"reason": "ok", "verdict": 1}])
        result = citation_accuracy(client, llm_config, "Réponse.", [{"text_span": "x"}])
        assert result["citation_accuracy"] == 1.0


class TestSufficiencyPrecision:
    def test_all_insufficient_verdicts_correct(self):
        records = [
            {"sufficiency_verdict": "insufficient", "should_abstain": True},
            {"sufficiency_verdict": "insufficient", "should_abstain": True},
        ]
        result = sufficiency_precision(records)
        assert result["sufficiency_precision"] == 1.0
        assert result["n_flagged_insufficient"] == 2

    def test_false_positive_lowers_precision(self):
        """Un verdict 'insufficient' sur une question in-corpus (devrait
        être answerable) est un faux positif."""
        records = [
            {"sufficiency_verdict": "insufficient", "should_abstain": True},
            {"sufficiency_verdict": "insufficient", "should_abstain": False},
        ]
        result = sufficiency_precision(records)
        assert result["sufficiency_precision"] == 0.5
        assert result["n_correct"] == 1

    def test_sufficient_and_partial_verdicts_ignored(self):
        """Ne compte que les verdicts 'insufficient' au dénominateur — c'est
        une mesure de précision sur cette classe précise, pas un score
        global toutes classes confondues."""
        records = [
            {"sufficiency_verdict": "sufficient", "should_abstain": False},
            {"sufficiency_verdict": "partial", "should_abstain": False},
            {"sufficiency_verdict": "insufficient", "should_abstain": True},
        ]
        result = sufficiency_precision(records)
        assert result["n_flagged_insufficient"] == 1
        assert result["sufficiency_precision"] == 1.0

    def test_no_insufficient_verdicts_returns_none_not_zero(self):
        records = [{"sufficiency_verdict": "sufficient", "should_abstain": False}]
        result = sufficiency_precision(records)
        assert result["sufficiency_precision"] is None
        assert result["n_flagged_insufficient"] == 0

    def test_empty_records_returns_none(self):
        result = sufficiency_precision([])
        assert result["sufficiency_precision"] is None
