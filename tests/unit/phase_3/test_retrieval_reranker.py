from src.retrieval.reranker import Reranker


def test_reranker_noop():
    r = Reranker()
    candidates = [
        {'chunk_id':'a','payload':{'text':'short'}},
        {'chunk_id':'b','payload':{'text':'a much longer text example'}}
    ]
    out = r.rerank('query', candidates, top_k=2)
    # fallback sorts by length ascending -> short first
    assert out[0]['chunk_id'] == 'a'


class TestWarmUp:
    """warm_up() doit forcer le chargement du modèle (et une inférence
    factice) explicitement, plutôt que de laisser la première requête
    utilisateur payer ce coût — voir Baseline C (audit Phase 5, p95 3-7s)."""

    def test_warm_up_loads_model_and_runs_dummy_inference(self, monkeypatch):
        r = Reranker()
        calls = []

        class _FakeModel:
            def compute_score(self, pairs, normalize=True):
                calls.append(pairs)
                return 0.5

        monkeypatch.setattr(r, "_get_model", lambda: _FakeModel())
        r.warm_up()
        assert len(calls) == 1

    def test_warm_up_does_not_crash_when_model_unavailable(self, monkeypatch):
        r = Reranker()
        monkeypatch.setattr(r, "_get_model", lambda: None)
        r.warm_up()  # ne doit pas lever

    def test_warm_up_does_not_crash_on_inference_failure(self, monkeypatch):
        r = Reranker()

        class _BrokenModel:
            def compute_score(self, pairs, normalize=True):
                raise RuntimeError("modèle non chargé correctement")

        monkeypatch.setattr(r, "_get_model", lambda: _BrokenModel())
        r.warm_up()  # ne doit pas lever
