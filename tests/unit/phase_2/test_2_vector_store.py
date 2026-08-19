"""Tests QdrantStore avec client in-memory."""

import pytest
import numpy as np
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import MatchAny, MatchValue

from src.embeddings.vector_store import QdrantStore
from src.ingestion.metadata import Chunk, ChunkMetadata
from src.dom.models import NodeType

pytestmark = pytest.mark.phase2


class TestBuildFilter:
    """``match: {"any": [...]}`` — comparer une question à plusieurs versions
    nommées explicitement plutôt qu'une seule (voir QueryPlanner filters,
    audit Precision@10)."""

    def test_match_value_single_condition(self):
        f = QdrantStore._build_filter({"key": "version_tag", "match": {"value": "v2323"}})
        assert isinstance(f.must[0].match, MatchValue)
        assert f.must[0].match.value == "v2323"

    def test_match_any_single_condition(self):
        f = QdrantStore._build_filter({"key": "version_tag", "match": {"any": ["legacy", "v2213"]}})
        assert isinstance(f.must[0].match, MatchAny)
        assert f.must[0].match.any == ["legacy", "v2213"]

    def test_match_any_inside_must_list(self):
        f = QdrantStore._build_filter(
            {"must": [{"key": "version_tag", "match": {"any": ["legacy", "v2213"]}}]}
        )
        assert isinstance(f.must[0].match, MatchAny)
        assert f.must[0].match.any == ["legacy", "v2213"]


class TestQdrantStore:
    def test_ensure_collection_creates_dense_and_sparse(self):
        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_collection")
        store.ensure_collection()
        info = client.get_collection("test_collection")
        assert "dense" in info.config.params.vectors
        assert "sparse" in info.config.params.sparse_vectors

    def test_upsert_and_search_dense(self):
        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_collection")
        store.ensure_collection()

        chunk = Chunk(
            source_id="test",
            node_ids=["n1"],
            text="hello world",
            raw_text="hello world",
            contextual_prefix="",
            metadata=ChunkMetadata(
                source_type="test",
                source_id="test",
                node_type=NodeType.PARAGRAPH,
                hierarchy_path="Doc > Sec",
                format_original="md",
            ),
            hierarchy_path="Doc > Sec",
        )
        chunk.content_hash = "abc123"

        from src.embeddings.bge_m3 import EmbeddingBatch
        emb = EmbeddingBatch(
            dense=np.array([[0.1] * 1024]),
            sparse=[{1: 0.5, 2: 0.3}],
        )

        store.upsert([chunk], emb)

        # Recherche dense
        results = store.search_dense(np.array([0.1] * 1024), k=5)
        assert len(results) == 1
        assert results[0].payload["source_id"] == "test"
        assert results[0].source == "dense"

    def test_idempotence_same_content_hash(self):
        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_collection")
        store.ensure_collection()

        chunk = Chunk(
            source_id="test",
            node_ids=["n1"],
            text="duplicate",
            raw_text="duplicate",
            contextual_prefix="",
            metadata=ChunkMetadata(
                source_type="test",
                source_id="test",
                node_type=NodeType.PARAGRAPH,
                hierarchy_path="Doc",
                format_original="md",
            ),
            hierarchy_path="Doc",
        )
        chunk.content_hash = "dup_hash"

        from src.embeddings.bge_m3 import EmbeddingBatch
        emb = EmbeddingBatch(
            dense=np.array([[0.2] * 1024]),
            sparse=[{1: 0.1}],
        )

        store.upsert([chunk], emb)
        store.upsert([chunk], emb)  # idempotent

        results = store.search_dense(np.array([0.2] * 1024), k=10)
        # Même content_hash -> écrasement, pas de doublon
        assert len(results) == 1