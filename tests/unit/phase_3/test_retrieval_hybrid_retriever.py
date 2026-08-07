"""Régression : le reranker doit réellement s'exécuter sur le chemin hybride
quand le pool de candidats fusionnés dépasse top_k.

Bug trouvé en testant la Phase 4 de bout en bout contre le vrai Qdrant :
HybridRetriever._hybrid_search() tronquait `fused` à `top_k` avant de
retourner, donc la condition `len(candidates) > top_k` dans retrieve() (qui
déclenche le reranking) n'était jamais vraie sur le chemin hybride — le
reranker ne s'exécutait donc jamais en pratique via HybridRetriever, malgré
les benchmarks scripts/measure_e2e_latency.py qui l'appellent en direct et
masquaient le problème."""

from uuid import uuid4

import numpy as np
import pytest

from src.embeddings.vector_store import RetrievalResult
from src.retrieval.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.phase3


def _make_results(n: int, prefix: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=uuid4(),
            score=1.0 - i * 0.01,
            source=prefix,
            payload={"text": f"{prefix} chunk {i}", "hierarchy_path": ""},
        )
        for i in range(n)
    ]


class TestHybridSearchTriggersReranking:
    def test_reranker_called_when_fused_pool_exceeds_top_k(self, monkeypatch):
        retriever = HybridRetriever(store=None, use_reranker=True)

        monkeypatch.setattr(
            retriever.dense_search, "search_dense", lambda *a, **k: _make_results(10, "dense")
        )
        monkeypatch.setattr(
            retriever.sparse_search, "search_sparse", lambda *a, **k: _make_results(10, "sparse")
        )

        rerank_calls = []

        def fake_rerank(query, candidates, top_k):
            rerank_calls.append(len(candidates))
            return candidates[:top_k]

        monkeypatch.setattr(retriever.reranker, "rerank", fake_rerank)

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "navigational", "entity": None, "filters": {}},
            top_k=5,
            k_dense=10,
            k_sparse=10,
        )

        assert rerank_calls, "reranker.rerank() was never called"
        assert rerank_calls[0] > 5, f"reranker only saw {rerank_calls[0]} candidates, pool was truncated before reranking"
        assert len(result["results"]) == 5

    def test_dense_only_fallback_also_not_pretruncated(self, monkeypatch):
        retriever = HybridRetriever(store=None, use_reranker=True)

        monkeypatch.setattr(
            retriever.dense_search, "search_dense", lambda *a, **k: _make_results(10, "dense")
        )
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        rerank_calls = []
        monkeypatch.setattr(
            retriever.reranker,
            "rerank",
            lambda query, candidates, top_k: rerank_calls.append(len(candidates)) or candidates[:top_k],
        )

        retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "navigational", "entity": None, "filters": {}},
            top_k=5,
            k_dense=10,
            k_sparse=10,
        )

        assert rerank_calls and rerank_calls[0] > 5
