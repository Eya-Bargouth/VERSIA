"""Tests unitaires — build_labelled_contexts (src/retrieval/context_fusion.py).
"""

import pytest

from src.retrieval.context_fusion import build_labelled_contexts

pytestmark = pytest.mark.phase3


class _FakeStore:
    """Duck-type minimal de QdrantStore.get_by_chunk_id."""

    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads
        self.calls: list[str] = []

    def get_by_chunk_id(self, chunk_id: str):
        self.calls.append(chunk_id)
        return self.payloads.get(chunk_id)


def _chunk(chunk_id: str, text: str, parent_chunk_id: str | None = None, **extra_payload) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "payload": {"parent_chunk_id": parent_chunk_id, "section_title": f"Section {chunk_id}", **extra_payload},
    }


class TestParentAlreadyInBatch:
    def test_parent_present_in_batch_is_included_without_store(self):
        parent = _chunk("parent-1", "Contenu du parent.")
        child = _chunk("child-1", "Contenu de l'enfant.", parent_chunk_id="parent-1")
        result = build_labelled_contexts([parent, child])
        assert "Parent Context" in result
        assert "Contenu du parent" in result


class TestParentMissingFromBatch:
    def test_missing_parent_omitted_without_store(self):
        """Comportement historique (avant #15) : sans store, un parent
        absent du lot reste silencieusement omis — pas de crash, mais pas
        d'expansion non plus."""
        child = _chunk("child-1", "Contenu de l'enfant.", parent_chunk_id="parent-absent")
        result = build_labelled_contexts([child])
        assert "Parent Context" not in result

    def test_missing_parent_fetched_explicitly_via_store(self):
        child = _chunk("child-1", "Contenu de l'enfant.", parent_chunk_id="parent-absent")
        store = _FakeStore({"parent-absent": {"text": "Contenu du parent récupéré.", "section_title": "Parent réel"}})
        result = build_labelled_contexts([child], store=store)
        assert "Parent Context" in result
        assert "Contenu du parent récupéré" in result
        assert store.calls == ["parent-absent"]

    def test_parent_not_found_in_store_degrades_without_crashing(self):
        child = _chunk("child-1", "Contenu de l'enfant.", parent_chunk_id="parent-absent")
        store = _FakeStore({})  # rien trouvé
        result = build_labelled_contexts([child], store=store)
        assert "Parent Context" not in result
        assert "Contenu de l'enfant" in result

    def test_same_parent_fetched_once_for_multiple_children(self):
        """Deux enfants partageant le même parent absent ne doivent
        déclencher qu'un seul point lookup Qdrant."""
        child_a = _chunk("child-a", "A.", parent_chunk_id="parent-absent")
        child_b = _chunk("child-b", "B.", parent_chunk_id="parent-absent")
        store = _FakeStore({"parent-absent": {"text": "Parent partagé.", "section_title": "P"}})
        build_labelled_contexts([child_a, child_b], store=store)
        assert store.calls == ["parent-absent"]

    def test_include_parent_false_skips_store_lookup_entirely(self):
        child = _chunk("child-1", "Contenu.", parent_chunk_id="parent-absent")
        store = _FakeStore({"parent-absent": {"text": "Ne devrait pas être appelé.", "section_title": "P"}})
        build_labelled_contexts([child], include_parent=False, store=store)
        assert store.calls == []

    def test_no_parent_id_never_calls_store(self):
        chunk = _chunk("solo", "Pas de parent.")
        store = _FakeStore({})
        build_labelled_contexts([chunk], store=store)
        assert store.calls == []
