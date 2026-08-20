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
            planner_output={"intent": "ambiguous", "entity": None, "filters": {}},
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
            planner_output={"intent": "ambiguous", "entity": None, "filters": {}},
            top_k=5,
            k_dense=10,
            k_sparse=10,
        )

        assert rerank_calls and rerank_calls[0] > 5


class TestVersionSiblingDedup:
    """Une source versionnée dont une section n'a pas changé d'une version à
    l'autre produit des chunks quasi identiques dans chaque fichier de
    version — ils ne doivent pas se marcher dessus dans le top-k."""

    def _version_result(self, hierarchy_path: str, version_order: int, score: float = 0.9, text: str | None = None):
        return RetrievalResult(
            chunk_id=uuid4(),
            score=score,
            source="dense",
            payload={
                "text": text if text is not None else f"contenu {hierarchy_path} v{version_order}",
                "hierarchy_path": hierarchy_path,
                "source_id": "stripe",
                "version_order": version_order,
            },
        )

    def test_keeps_only_highest_version_order_per_hierarchy_path(self, monkeypatch):
        retriever = HybridRetriever(store=None, use_reranker=False)
        siblings = [
            self._version_result("paths./v1/apps/secrets.get", 1),
            self._version_result("paths./v1/apps/secrets.get", 3),
            self._version_result("paths./v1/apps/secrets.get", 2),
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: siblings)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        kept = result["results"]
        assert len(kept) == 1
        assert kept[0]["payload"]["version_order"] == 3

    def test_comparative_intent_keeps_all_versions(self, monkeypatch):
        retriever = HybridRetriever(store=None, use_reranker=False)
        siblings = [
            self._version_result("paths./v1/apps/secrets.get", 1),
            self._version_result("paths./v1/apps/secrets.get", 2),
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: siblings)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "comparative", "entity": None, "filters": {"version_tags": ["tagA", "tagB"]}},
            top_k=10,
        )

        assert len(result["results"]) == 2

    def test_comparative_intent_collapses_identical_content(self, monkeypatch):
        """Contenu inchangé d'une version à l'autre : même en intent
        comparatif, comparer 4 copies identiques n'apporte rien — seule la
        plus récente est gardée."""
        retriever = HybridRetriever(store=None, use_reranker=False)
        same_text = "Le paramètre symbol est requis."
        siblings = [
            self._version_result("paths./v1/apps/secrets.get", 1, text=same_text),
            self._version_result("paths./v1/apps/secrets.get", 2, text=same_text),
            self._version_result("paths./v1/apps/secrets.get", 3, text=same_text),
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: siblings)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "comparative", "entity": None, "filters": {"version_tags": ["tagA", "tagB"]}},
            top_k=10,
        )

        kept = result["results"]
        assert len(kept) == 1
        assert kept[0]["payload"]["version_order"] == 3

    def test_comparative_intent_keeps_distinct_drops_identical(self, monkeypatch):
        """Groupe mixte : 2 versions au contenu réellement différent (gardées
        toutes les deux) + 1 version dont le texte duplique l'une d'elles
        (éliminée)."""
        retriever = HybridRetriever(store=None, use_reranker=False)
        siblings = [
            self._version_result("paths./v1/x.get", 1, text="ancien comportement"),
            self._version_result("paths./v1/x.get", 2, text="nouveau comportement"),
            self._version_result("paths./v1/x.get", 3, text="nouveau comportement"),  # doublon de v2
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: siblings)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "comparative", "entity": None, "filters": {"version_tags": ["tagA", "tagB"]}},
            top_k=10,
        )

        kept_orders = sorted(r["payload"]["version_order"] for r in result["results"])
        assert kept_orders == [1, 3]  # v3 gagne sur v2 (même texte, plus récente), v1 distinct gardé

    def test_comparative_without_version_signal_leaves_candidates_untouched(self, monkeypatch):
        """Régression : une question comparative qui compare des CONCEPTS
        (ex. "différence entre l'authentification TLS et l'API Stripe"), pas
        des versions — aucun tag détecté par le planner (filters vide) — ne
        doit subir aucun dédup. Sans ce garde-fou, une version "jugée
        identique en contenu" pouvait être éliminée alors qu'elle était
        justement celle attendue par la question (mesuré : Recall@10
        multi_source/ambiguous 0.6→0.2)."""
        retriever = HybridRetriever(store=None, use_reranker=False)
        same_text = "contenu inchangé"
        siblings = [
            self._version_result("paths./v1/x.get", 1, text=same_text),
            self._version_result("paths./v1/x.get", 3, text=same_text),
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: siblings)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "comparative", "entity": None, "filters": {}},
            top_k=10,
        )

        assert len(result["results"]) == 2  # aucun candidat éliminé

    def test_unversioned_chunks_never_merged(self, monkeypatch):
        retriever = HybridRetriever(store=None, use_reranker=False)
        unversioned = [
            RetrievalResult(
                chunk_id=uuid4(), score=0.9, source="dense",
                payload={"text": "a", "hierarchy_path": "same.path", "source_id": "owasp_cheatsheets"},
            ),
            RetrievalResult(
                chunk_id=uuid4(), score=0.8, source="dense",
                payload={"text": "b", "hierarchy_path": "same.path", "source_id": "owasp_cheatsheets"},
            ),
        ]
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: unversioned)
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        assert len(result["results"]) == 2


class _FakeStoreWithSiblings:
    """Store minimal exposant seulement get_by_parent_path, pour tester
    _fetch_missing_siblings sans vrai Qdrant."""

    def __init__(self, siblings_by_group: dict):
        self.siblings_by_group = siblings_by_group
        self.calls: list[tuple] = []

    def get_by_parent_path(self, parent_path, source_id=None, version_tag=None, limit=50):
        self.calls.append((parent_path, source_id, version_tag))
        return self.siblings_by_group.get((parent_path, source_id, version_tag), [])


class TestSiblingExpansion:
    """Un chunk retrouvé appartenant à une liste source (parent_path défini)
    doit se voir compléter par ses frères manquants — cas réel : le
    paramètre requis d'un endpoint absent du top-10 alors que des paramètres
    optionnels du même endpoint y figuraient."""

    def test_missing_siblings_are_fetched_and_added(self, monkeypatch):
        anchor = RetrievalResult(
            chunk_id=uuid4(), score=0.8, source="dense",
            payload={"text": "limit param", "hierarchy_path": "x.parameters[1]",
                     "parent_path": "x.parameters", "source_id": "stripe", "version_tag": "v1"},
        )
        missing_sibling_payload = {
            "chunk_id": "sib-1", "text": "query param (required)",
            "hierarchy_path": "x.parameters[0]", "parent_path": "x.parameters",
            "source_id": "stripe", "version_tag": "v1",
        }
        fake_store = _FakeStoreWithSiblings({
            ("x.parameters", "stripe", "v1"): [missing_sibling_payload],
        })
        retriever = HybridRetriever(store=fake_store, use_reranker=False)
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: [anchor])
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        chunk_ids = {r["chunk_id"] for r in result["results"]}
        assert "sib-1" in chunk_ids
        assert len(result["results"]) == 2
        added = next(r for r in result["results"] if r["chunk_id"] == "sib-1")
        assert added["score"] == 0.0

    def test_already_present_sibling_not_duplicated(self, monkeypatch):
        chunk_id_a = str(uuid4())
        anchor = RetrievalResult(
            chunk_id=chunk_id_a, score=0.8, source="dense",
            payload={"text": "a", "hierarchy_path": "x.parameters[0]",
                     "parent_path": "x.parameters", "source_id": "stripe", "version_tag": "v1"},
        )
        fake_store = _FakeStoreWithSiblings({
            ("x.parameters", "stripe", "v1"): [
                {"chunk_id": chunk_id_a, "text": "a", "hierarchy_path": "x.parameters[0]",
                 "parent_path": "x.parameters", "source_id": "stripe", "version_tag": "v1"},
            ],
        })
        retriever = HybridRetriever(store=fake_store, use_reranker=False)
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: [anchor])
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        assert len(result["results"]) == 1

    def test_no_parent_path_skips_lookup_entirely(self, monkeypatch):
        anchor = RetrievalResult(
            chunk_id=uuid4(), score=0.8, source="dense",
            payload={"text": "a", "hierarchy_path": "x.summary", "source_id": "stripe"},
        )
        fake_store = _FakeStoreWithSiblings({})
        retriever = HybridRetriever(store=fake_store, use_reranker=False)
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: [anchor])
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        assert fake_store.calls == []

    def test_no_store_does_not_crash(self, monkeypatch):
        anchor = RetrievalResult(
            chunk_id=uuid4(), score=0.8, source="dense",
            payload={"text": "a", "hierarchy_path": "x.parameters[0]",
                     "parent_path": "x.parameters", "source_id": "stripe"},
        )
        retriever = HybridRetriever(store=None, use_reranker=False)
        monkeypatch.setattr(retriever.dense_search, "search_dense", lambda *a, **k: [anchor])
        monkeypatch.setattr(retriever.sparse_search, "search_sparse", lambda *a, **k: [])

        result = retriever.retrieve(
            "query",
            query_embedding=np.zeros(1024, dtype=np.float32),
            query_sparse={0: 1.0},
            planner_output={"intent": "factual", "entity": None, "filters": {}},
            top_k=10,
        )

        assert len(result["results"]) == 1
