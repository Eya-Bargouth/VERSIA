"""Tests AbstentionGate — 3 zones, seuils provisoires, pénalité planner."""

import pytest

from src.reliability.abstention import AbstentionGate
from src.reliability.sufficiency import SufficiencyVerdict

pytestmark = pytest.mark.phase4


def _verdict(v: str, confidence: float = 0.9) -> SufficiencyVerdict:
    return SufficiencyVerdict(verdict=v, confidence=confidence, method="llm_local")


class TestAbstentionGate:
    def test_high_scores_yield_correct_zone(self):
        gate = AbstentionGate()
        result = gate.evaluate(
            reranker_scores=[0.9, 0.85],
            generation_confidence=0.9,
            sufficiency_verdict=_verdict("sufficient", 0.9),
        )
        assert result.zone == "correct"
        assert result.action == "answer"

    def test_low_scores_yield_incorrect_zone(self):
        gate = AbstentionGate()
        result = gate.evaluate(
            reranker_scores=[0.05, 0.1],
            generation_confidence=0.1,
            sufficiency_verdict=_verdict("insufficient", 0.1),
        )
        assert result.zone == "incorrect"
        assert result.action == "abstain"

    def test_mid_scores_yield_ambiguous_zone(self):
        gate = AbstentionGate()
        result = gate.evaluate(
            reranker_scores=[0.5, 0.5],
            generation_confidence=0.5,
            sufficiency_verdict=_verdict("partial", 0.5),
        )
        assert result.zone == "ambiguous"
        assert result.action == "answer_with_caveat"

    def test_planner_low_confidence_pulls_score_down(self):
        gate = AbstentionGate()
        kwargs = dict(
            reranker_scores=[0.9, 0.85],
            generation_confidence=0.9,
            sufficiency_verdict=_verdict("sufficient", 0.9),
        )
        with_flag = gate.evaluate(**kwargs, planner_confidence="low")
        without_flag = gate.evaluate(**kwargs)
        assert with_flag.combined_score < without_flag.combined_score

    def test_empty_reranker_scores_treated_as_zero(self):
        gate = AbstentionGate()
        result = gate.evaluate(
            reranker_scores=[],
            generation_confidence=0.9,
            sufficiency_verdict=_verdict("sufficient", 0.9),
        )
        # 0.3*0 + 0.3*0.9 + 0.3*1.0 + 0.1*0.9 = 0.66 -> ambiguous, pas correct
        assert result.zone == "ambiguous"

    def test_out_of_range_reranker_score_is_clipped_not_crashing(self):
        gate = AbstentionGate()
        result = gate.evaluate(
            reranker_scores=[5.0, -3.0],  # repli heuristique non borné
            generation_confidence=0.5,
            sufficiency_verdict=_verdict("partial", 0.5),
        )
        assert 0.0 <= result.combined_score <= 1.0

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(ValueError):
            AbstentionGate(threshold_low=0.8, threshold_high=0.2)
