"""Tests unitaires — métriques retrieval (Recall@k, Precision@k, MRR, nDCG)."""

import pytest

from src.evaluation.retrieval_metrics import (
    evaluate_query,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.phase5


def _result(source_path: str) -> dict:
    return {"payload": {"source_path": source_path}}


class TestIsRelevantMatching:
    def test_recall_perfect_when_all_sources_found(self):
        results = [_result("raw/a.yaml"), _result("raw/b.yaml")]
        assert recall_at_k(results, ["raw/a.yaml", "raw/b.yaml"], k=2) == 1.0

    def test_recall_partial(self):
        results = [_result("raw/a.yaml"), _result("raw/other.yaml")]
        assert recall_at_k(results, ["raw/a.yaml", "raw/b.yaml"], k=2) == 0.5

    def test_recall_no_expected_sources_is_trivially_perfect(self):
        assert recall_at_k([_result("raw/a.yaml")], [], k=5) == 1.0

    def test_recall_ignores_results_beyond_k(self):
        results = [_result("raw/other.yaml"), _result("raw/other2.yaml"), _result("raw/a.yaml")]
        assert recall_at_k(results, ["raw/a.yaml"], k=2) == 0.0
        assert recall_at_k(results, ["raw/a.yaml"], k=3) == 1.0


class TestPrecisionAtK:
    def test_precision_all_relevant(self):
        results = [_result("raw/a.yaml"), _result("raw/a.yaml")]
        assert precision_at_k(results, ["raw/a.yaml"], k=2) == 1.0

    def test_precision_half_relevant(self):
        results = [_result("raw/a.yaml"), _result("raw/other.yaml")]
        assert precision_at_k(results, ["raw/a.yaml"], k=2) == 0.5

    def test_precision_empty_results(self):
        assert precision_at_k([], ["raw/a.yaml"], k=5) == 0.0


class TestMRR:
    def test_mrr_first_result_relevant(self):
        results = [_result("raw/a.yaml"), _result("raw/other.yaml")]
        assert mrr(results, ["raw/a.yaml"]) == 1.0

    def test_mrr_third_result_relevant(self):
        results = [_result("raw/x.yaml"), _result("raw/y.yaml"), _result("raw/a.yaml")]
        assert mrr(results, ["raw/a.yaml"]) == pytest.approx(1 / 3)

    def test_mrr_no_relevant_result(self):
        results = [_result("raw/x.yaml")]
        assert mrr(results, ["raw/a.yaml"]) == 0.0


class TestNDCG:
    def test_ndcg_perfect_ordering(self):
        results = [_result("raw/a.yaml"), _result("raw/b.yaml")]
        assert ndcg_at_k(results, ["raw/a.yaml", "raw/b.yaml"], k=2) == pytest.approx(1.0)

    def test_ndcg_worse_when_relevant_ranked_lower(self):
        perfect = [_result("raw/a.yaml"), _result("raw/other.yaml")]
        worse = [_result("raw/other.yaml"), _result("raw/a.yaml")]
        assert ndcg_at_k(worse, ["raw/a.yaml"], k=2) < ndcg_at_k(perfect, ["raw/a.yaml"], k=2)

    def test_ndcg_zero_when_no_expected_sources_found_but_k_positive(self):
        results = [_result("raw/other.yaml")]
        assert ndcg_at_k(results, ["raw/a.yaml"], k=1) == 0.0

    def test_ndcg_stays_bounded_when_one_source_yields_many_relevant_chunks(self):
        """Régression : une seule source attendue (ex. version_conflict avec
        2 fichiers) peut légitimement fournir plusieurs chunks pertinents
        dans le top-k — l'IDCG ne doit pas être calculé à partir du nombre
        de sources attendues (bug trouvé en pratique : nDCG > 1.0)."""
        results = [_result("raw/a.yaml")] * 5
        score = ndcg_at_k(results, ["raw/a.yaml"], k=5)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(1.0)  # tout pertinent -> ordre déjà idéal

    def test_ndcg_never_exceeds_one_across_many_relevance_patterns(self):
        import itertools
        for pattern in itertools.product([True, False], repeat=4):
            results = [_result("raw/a.yaml") if rel else _result("raw/x.yaml") for rel in pattern]
            score = ndcg_at_k(results, ["raw/a.yaml"], k=4)
            assert 0.0 <= score <= 1.0 + 1e-9


class TestPathMatchingRobustness:
    def test_matches_despite_different_path_separators(self):
        results = [_result("raw\\specs-api\\stripe\\spec3-v2323.yaml")]
        assert recall_at_k(results, ["raw/specs-api/stripe/spec3-v2323.yaml"], k=1) == 1.0

    def test_matches_by_suffix_despite_prefix_difference(self):
        results = [_result("C:/project/raw/specs-api/stripe/spec3-v2323.yaml")]
        assert recall_at_k(results, ["raw/specs-api/stripe/spec3-v2323.yaml"], k=1) == 1.0

    def test_does_not_match_unrelated_file(self):
        results = [_result("raw/specs-api/binance/spot_api.yaml")]
        assert recall_at_k(results, ["raw/specs-api/stripe/spec3-v2323.yaml"], k=1) == 0.0


class TestEvaluateQuery:
    def test_returns_all_metrics_for_each_k(self):
        results = [_result("raw/a.yaml")]
        metrics = evaluate_query(results, ["raw/a.yaml"], k_values=[1, 5])
        assert set(metrics.keys()) == {
            "mrr", "recall@1", "precision@1", "ndcg@1", "recall@5", "precision@5", "ndcg@5",
        }
        assert metrics["recall@1"] == 1.0
