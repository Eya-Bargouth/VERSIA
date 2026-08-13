"""Tests unitaires — Hallucination Rate."""

import pytest
from pydantic import BaseModel

from src.evaluation.hallucination import hallucination_rate

pytestmark = pytest.mark.phase5


class _FakeCitation(BaseModel):
    support_level: str


class _FakeResult(BaseModel):
    citations: list[_FakeCitation]


def _result(*support_levels: str) -> _FakeResult:
    return _FakeResult(citations=[_FakeCitation(support_level=s) for s in support_levels])


def test_no_hallucination_when_all_fully_supported():
    results = [_result("fully_supported"), _result("fully_supported", "partially_supported")]
    out = hallucination_rate(results)
    assert out["hallucination_rate"] == 0.0
    assert out["n_answered"] == 2
    assert out["n_hallucinated"] == 0


def test_hallucination_detected_on_no_support_citation():
    results = [_result("fully_supported"), _result("no_support")]
    out = hallucination_rate(results)
    assert out["hallucination_rate"] == 0.5
    assert out["n_hallucinated"] == 1


def test_abstained_answers_excluded_from_denominator():
    """Une abstention (citations=[]) ne compte ni comme halluciné ni comme
    non-halluciné — elle est exclue du dénominateur, pas assimilée à un
    succès."""
    results = [_result(), _result("no_support")]
    out = hallucination_rate(results)
    assert out["n_abstained"] == 1
    assert out["n_answered"] == 1
    assert out["hallucination_rate"] == 1.0


def test_empty_results_list_returns_zero_not_error():
    out = hallucination_rate([])
    assert out["hallucination_rate"] == 0.0
    assert out["n_total"] == 0
