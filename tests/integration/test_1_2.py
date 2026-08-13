"""Test d'intégration cumulatif Phase 1 + Phase 2."""

import shutil

import pytest
import numpy as np
from qdrant_client import QdrantClient

from src.embeddings.bge_m3 import EmbeddingBatch
from src.embeddings.vector_store import QdrantStore
from src.ingestion.pipeline import run_ingestion

pytestmark = pytest.mark.phase2


class MockBGEEmbedder:
    """Embedder factice pour les tests d'intégration — ne charge pas BGE-M3."""

    def __init__(self):
        self.counter = 0

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        n = len(texts)
        dense = np.random.rand(n, 1024).astype(np.float32)
        # Normaliser pour cosine similarity
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / norms
        sparse = [{i: 0.1 for i in range(min(10, len(t)))} for t in texts]
        return EmbeddingBatch(dense=dense, sparse=sparse)

    def embed_query(self, query: str):
        batch = self.embed([query])
        return batch.dense[0], batch.sparse[0]


class TestPhase1Phase2Integration:
    def test_full_pipeline_from_raw_dir(self, fixtures_dir, tmp_path):
        """Ingestion complète d'une fixture, découverte 100% automatique
        (aucun manifeste — le dossier devient la source, voir
        src.ingestion.pipeline.discover_sources)."""
        raw_dir = tmp_path / "raw"
        source_dir = raw_dir / "test_api_integration"
        source_dir.mkdir(parents=True)
        shutil.copy(fixtures_dir / "sample_openapi.yaml", source_dir / "sample_openapi.yaml")

        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_1_2_b")
        store.ensure_collection()
        embedder = MockBGEEmbedder()

        report = run_ingestion(raw_dir=raw_dir, store=store, embedder=embedder)

        assert report.total_documents >= 1
        assert report.total_chunks >= 1
        assert report.by_source["test_api_integration"]["chunks"] >= 1

        # Vérifier retrieval dense
        dense_vec, _ = embedder.embed_query("Create a new order")
        results = store.search_dense(dense_vec, k=5)
        assert len(results) >= 1
        assert any("orders" in (r.payload.get("text") or "") for r in results)

    def test_idempotence_pipeline(self, fixtures_dir, tmp_path):
        """Relancer le pipeline ne crée pas de doublons."""
        raw_dir = tmp_path / "raw"
        source_dir = raw_dir / "test_idem"
        source_dir.mkdir(parents=True)
        shutil.copy(fixtures_dir / "sample_markdown.md", source_dir / "sample_markdown.md")

        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_idem")
        store.ensure_collection()
        embedder = MockBGEEmbedder()

        report1 = run_ingestion(raw_dir=raw_dir, store=store, embedder=embedder)
        chunks_first = report1.total_chunks

        report2 = run_ingestion(raw_dir=raw_dir, store=store, embedder=embedder)
        chunks_second = report2.total_chunks

        # Idempotence : même nombre de chunks logiques (même content_hash -> écrasement)
        assert chunks_first == chunks_second

        # Vérifier qu'on ne retrouve pas de doublons physiques
        dense_vec, _ = embedder.embed_query("password storage")
        results = store.search_dense(dense_vec, k=50)
        texts = [r.payload.get("text", "") for r in results]
        # Pas de textes identiques en double
        assert len(texts) == len(set(texts))
