from uuid import uuid4

import numpy as np
import pytest

from src.embeddings.vector_store import RetrievalResult
from src.ingestion.version_diff import DiffReport
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


class _FakeStoreWithVersionCounterparts:
    """Store minimal exposant seulement get_by_hierarchy_path, pour tester
    _complete_version_pairs sans vrai Qdrant."""

    def __init__(self, payload_by_key: dict):
        self.payload_by_key = payload_by_key
        self.calls: list[tuple] = []

    def get_by_hierarchy_path(self, hierarchy_path, source_id=None, version_tag=None, limit=5):
        self.calls.append((hierarchy_path, source_id, version_tag))
        payload = self.payload_by_key.get((hierarchy_path, source_id, version_tag))
        return [payload] if payload else []


class TestVersionPairCompletion:
    """Question comparative (diff_report chargé) : si un candidat retrouvé
    n'a que l'un des deux côtés de la comparaison, l'autre côté doit être
    complété directement — cas réel mesuré : context_precision=0.0
    récurrent sur version_conflict, faute d'avoir les deux versions dans le
    contexte transmis au générateur."""

    def _diff_report(self) -> DiffReport:
        return DiffReport(source_id="plaid", version_from="1.19.5-beta", version_to="1.20.6", changes=[])

    def test_missing_counterpart_is_fetched_and_added(self):
        candidates = [
            {
                "chunk_id": "a",
                "score": 0.8,
                "text": "old description",
                "payload": {"hierarchy_path": "schemas.Item.field", "source_id": "plaid", "version_tag": "1.19.5-beta"},
            }
        ]
        counterpart_payload = {
            "chunk_id": "b",
            "text": "new description",
            "hierarchy_path": "schemas.Item.field",
            "source_id": "plaid",
            "version_tag": "1.20.6",
        }
        fake_store = _FakeStoreWithVersionCounterparts({
            ("schemas.Item.field", "plaid", "1.20.6"): counterpart_payload,
        })
        retriever = HybridRetriever(store=fake_store, use_reranker=False)

        result = retriever._complete_version_pairs(candidates, self._diff_report())

        chunk_ids = {c["chunk_id"] for c in result}
        assert chunk_ids == {"a", "b"}
        added = next(c for c in result if c["chunk_id"] == "b")
        assert added["score"] == 0.0
        assert fake_store.calls == [("schemas.Item.field", "plaid", "1.20.6")]

    def test_both_sides_already_present_no_fetch(self):
        candidates = [
            {"chunk_id": "a", "score": 0.8, "text": "old", "payload": {"hierarchy_path": "p", "source_id": "plaid", "version_tag": "1.19.5-beta"}},
            {"chunk_id": "b", "score": 0.7, "text": "new", "payload": {"hierarchy_path": "p", "source_id": "plaid", "version_tag": "1.20.6"}},
        ]
        fake_store = _FakeStoreWithVersionCounterparts({})
        retriever = HybridRetriever(store=fake_store, use_reranker=False)

        result = retriever._complete_version_pairs(candidates, self._diff_report())

        assert len(result) == 2
        assert fake_store.calls == []

    def test_candidate_outside_version_pair_ignored(self):
        """Un chunk d'une version non concernée par CE diff (ex. 1.5.0-beta,
        ni version_from ni version_to) ne doit déclencher aucune recherche."""
        candidates = [
            {"chunk_id": "a", "score": 0.8, "text": "x", "payload": {"hierarchy_path": "p", "source_id": "plaid", "version_tag": "1.5.0-beta"}},
        ]
        fake_store = _FakeStoreWithVersionCounterparts({})
        retriever = HybridRetriever(store=fake_store, use_reranker=False)

        result = retriever._complete_version_pairs(candidates, self._diff_report())

        assert len(result) == 1
        assert fake_store.calls == []

    def test_no_store_returns_candidates_unchanged(self):
        candidates = [
            {"chunk_id": "a", "score": 0.8, "text": "x", "payload": {"hierarchy_path": "p", "source_id": "plaid", "version_tag": "1.19.5-beta"}},
        ]
        retriever = HybridRetriever(store=None, use_reranker=False)

        result = retriever._complete_version_pairs(candidates, self._diff_report())

        assert result == candidates


class _FakeStoreWithChangeSearch:
    """Store minimal exposant search_change_nodes + get_by_hierarchy_path,
    pour tester _search_relevant_changes/_fetch_content_for_changes sans
    vrai Qdrant."""

    def __init__(self, change_hits: list, content_by_key: dict | None = None):
        self.change_hits = change_hits
        self.content_by_key = content_by_key or {}
        self.search_calls: list = []
        self.fetch_calls: list = []

    def search_change_nodes(self, query_embedding, source_id, version_from, version_to, k=10):
        self.search_calls.append((source_id, version_from, version_to, k))
        return self.change_hits

    def get_by_hierarchy_path(self, hierarchy_path, source_id=None, version_tag=None, limit=5):
        self.fetch_calls.append((hierarchy_path, source_id, version_tag))
        payload = self.content_by_key.get((hierarchy_path, source_id, version_tag))
        return [payload] if payload else []


class _FakeChangeHit:
    def __init__(self, payload):
        self.payload = payload


class TestChangeNodeRetrieval:
    """Recherche sémantique sur les nœuds Change (VersionRAG §4.2) : une
    fois le couple de versions résolu, la question en langage naturel est
    comparée aux descriptions de changements indexées plutôt qu'un simple
    matching de tags — puis le contenu des deux côtés est récupéré
    directement par hierarchy_path, sans dépendre du hasard du retrieval
    vectoriel générique."""

    def _diff_report(self):
        return DiffReport(source_id="plaid", version_from="1.19.5-beta", version_to="1.20.6", changes=[])

    def test_search_relevant_changes_delegates_to_store(self):
        hits = [_FakeChangeHit({"hierarchy_path": "schemas.Item.field"})]
        fake_store = _FakeStoreWithChangeSearch(hits)
        retriever = HybridRetriever(store=fake_store, use_reranker=False)

        result = retriever._search_relevant_changes(np.zeros(1024, dtype=np.float32), self._diff_report())

        assert result == hits
        assert fake_store.search_calls == [("plaid", "1.19.5-beta", "1.20.6", 5)]

    def test_search_relevant_changes_no_store_returns_empty(self):
        retriever = HybridRetriever(store=None, use_reranker=False)
        result = retriever._search_relevant_changes(np.zeros(1024, dtype=np.float32), self._diff_report())
        assert result == []

    def test_search_relevant_changes_store_error_degrades_gracefully(self):
        class _BrokenStore:
            def search_change_nodes(self, *a, **k):
                raise RuntimeError("qdrant down")

        retriever = HybridRetriever(store=_BrokenStore(), use_reranker=False)
        result = retriever._search_relevant_changes(np.zeros(1024, dtype=np.float32), self._diff_report())
        assert result == []

    def test_fetch_content_for_changes_adds_both_versions(self):
        hits = [_FakeChangeHit({"hierarchy_path": "schemas.Item.field"})]
        content = {
            ("schemas.Item.field", "plaid", "1.19.5-beta"): {"chunk_id": "old", "text": "old text"},
            ("schemas.Item.field", "plaid", "1.20.6"): {"chunk_id": "new", "text": "new text"},
        }
        fake_store = _FakeStoreWithChangeSearch(hits, content)
        retriever = HybridRetriever(store=fake_store, use_reranker=False)

        result = retriever._fetch_content_for_changes([], hits, self._diff_report())

        chunk_ids = {c["chunk_id"] for c in result}
        assert chunk_ids == {"old", "new"}
        assert all(c["score"] == 0.0 for c in result)

    def test_fetch_content_for_changes_skips_already_present(self):
        hits = [_FakeChangeHit({"hierarchy_path": "schemas.Item.field"})]
        content = {("schemas.Item.field", "plaid", "1.20.6"): {"chunk_id": "new", "text": "new text"}}
        fake_store = _FakeStoreWithChangeSearch(hits, content)
        retriever = HybridRetriever(store=fake_store, use_reranker=False)
        existing = [{"chunk_id": "old", "score": 0.5, "text": "already here", "payload": {}}]

        result = retriever._fetch_content_for_changes(existing, hits, self._diff_report())

        assert len(result) == 2
        assert {c["chunk_id"] for c in result} == {"old", "new"}

    def test_fetch_content_for_changes_no_hits_returns_candidates_unchanged(self):
        retriever = HybridRetriever(store=_FakeStoreWithChangeSearch([]), use_reranker=False)
        candidates = [{"chunk_id": "a", "score": 0.5, "text": "x", "payload": {}}]
        result = retriever._fetch_content_for_changes(candidates, [], self._diff_report())
        assert result == candidates
