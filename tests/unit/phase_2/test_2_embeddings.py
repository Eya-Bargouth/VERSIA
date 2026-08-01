"""Tests BGEEmbedder (mocké)."""

import pytest
import numpy as np

from src.embeddings.bge_m3 import BGEEmbedder, EmbeddingBatch

pytestmark = pytest.mark.phase2


class TestBGEEmbedderMocked:
    def test_embed_returns_correct_shapes(self, monkeypatch):
        embedder = BGEEmbedder(model_name="fake-model", device="cpu")
        # Mock _load_model et l'encodeur interne
        fake_dense = np.random.rand(2, 1024).astype(np.float32)
        fake_sparse = [{1: 0.5}, {2: 0.3}]

        def fake_encode(sentences, **kwargs):
            return {
                "dense_vecs": fake_dense,
                "lexical_weights": fake_sparse,
            }

        embedder._model = type("FakeModel", (), {"encode": staticmethod(fake_encode)})()
        embedder._load_model = lambda: None

        batch = embedder.embed(["hello world", "test query"])
        assert batch.dense.shape == (2, 1024)
        assert len(batch.sparse) == 2

    def test_embed_query_returns_single_vectors(self, monkeypatch):
        embedder = BGEEmbedder(model_name="fake-model", device="cpu")
        fake_dense = np.random.rand(1, 1024).astype(np.float32)
        fake_sparse = [{1: 0.5}]

        def fake_encode(sentences, **kwargs):
            return {
                "dense_vecs": fake_dense,
                "lexical_weights": fake_sparse,
            }

        embedder._model = type("FakeModel", (), {"encode": staticmethod(fake_encode)})()
        embedder._load_model = lambda: None

        dense, sparse = embedder.embed_query("query")
        assert dense.shape == (1024,)
        assert isinstance(sparse, dict)