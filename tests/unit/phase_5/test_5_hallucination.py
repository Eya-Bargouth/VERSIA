"""Tests unitaires — Hallucination Rate.

Dérivée du score Faithfulness du juge indépendant (ragas_eval.faithfulness),
pas du support_level auto-déclaré par le générateur — voir docstring de
src/evaluation/hallucination.py pour le biais structurel qui a motivé ce
changement (prompt interdisait le label "no_support", rate figé à 0.0)."""

import pytest
from pydantic import BaseModel

from src.evaluation.hallucination import hallucination_rate

pytestmark = pytest.mark.phase5


class _FakeResult(BaseModel):
    citations: list[str]  # juste besoin d'une liste non vide/vide, contenu sans importance
    faithfulness: float


def _result(faithfulness: float, has_citations: bool = True) -> _FakeResult:
    return _FakeResult(citations=["cit_001"] if has_citations else [], faithfulness=faithfulness)


def test_no_hallucination_when_faithfulness_above_threshold():
    results = [_result(1.0), _result(0.8)]
    out = hallucination_rate(results)
    assert out["hallucination_rate"] == 0.0
    assert out["n_answered"] == 2
    assert out["n_hallucinated"] == 0


def test_hallucination_detected_below_threshold():
    results = [_result(0.9), _result(0.2)]
    out = hallucination_rate(results)
    assert out["hallucination_rate"] == 0.5
    assert out["n_hallucinated"] == 1


def test_abstained_answers_excluded_from_denominator():
    """Une abstention (citations=[]) ne compte ni comme halluciné ni comme
    non-halluciné — elle est exclue du dénominateur, pas assimilée à un
    succès."""
    results = [_result(0.0, has_citations=False), _result(0.1)]
    out = hallucination_rate(results)
    assert out["n_abstained"] == 1
    assert out["n_answered"] == 1
    assert out["hallucination_rate"] == 1.0


def test_empty_results_list_returns_zero_not_error():
    out = hallucination_rate([])
    assert out["hallucination_rate"] == 0.0
    assert out["n_total"] == 0


def test_threshold_is_configurable():
    results = [_result(0.6)]
    assert hallucination_rate(results, threshold=0.5)["n_hallucinated"] == 0
    assert hallucination_rate(results, threshold=0.7)["n_hallucinated"] == 1


def test_boundary_score_equal_to_threshold_is_not_hallucinated():
    """< threshold, pas <= — un score exactement au seuil est encore
    considéré fidèle, cohérent avec la sémantique "sous le seuil = halluciné"."""
    results = [_result(0.5)]
    assert hallucination_rate(results, threshold=0.5)["n_hallucinated"] == 0
