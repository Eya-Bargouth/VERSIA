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

    def test_get_by_parent_path_finds_siblings_same_version_only(self):
        """3 éléments d'une même liste source (ex. paramètres d'un endpoint,
        hierarchy_path se terminant par [0]/[1]/[2]) dans la version v1, +
        1 élément homonyme dans la version v2 — get_by_parent_path ne doit
        renvoyer que les frères de la MÊME version que celle demandée."""
        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_collection")
        store.ensure_collection()

        from src.embeddings.bge_m3 import EmbeddingBatch

        def _param_chunk(idx: int, version_tag: str, content_hash: str):
            return Chunk(
                source_id="stripe",
                node_ids=[f"n{idx}"],
                text=f"param {idx}",
                raw_text=f"param {idx}",
                contextual_prefix="",
                version_tag=version_tag,
                metadata=ChunkMetadata(
                    source_type="test", source_id="stripe", node_type=NodeType.DOCUMENT,
                    hierarchy_path=f"paths./v1/x.get.parameters[{idx}]", format_original="yaml",
                ),
                hierarchy_path=f"paths./v1/x.get.parameters[{idx}]",
                content_hash=content_hash,
            )

        chunks = [
            _param_chunk(0, "v1", "h0v1"),
            _param_chunk(1, "v1", "h1v1"),
            _param_chunk(2, "v1", "h2v1"),
            _param_chunk(0, "v2", "h0v2"),
        ]
        emb = EmbeddingBatch(dense=np.array([[0.3] * 1024] * 4), sparse=[{1: 0.1}] * 4)
        store.upsert(chunks, emb)

        siblings = store.get_by_parent_path("paths./v1/x.get.parameters", source_id="stripe", version_tag="v1")
        assert len(siblings) == 3
        assert {s["hierarchy_path"] for s in siblings} == {
            "paths./v1/x.get.parameters[0]",
            "paths./v1/x.get.parameters[1]",
            "paths./v1/x.get.parameters[2]",
        }

    def test_ensure_parent_path_index_is_idempotent(self):
        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="test_collection")
        store.ensure_collection()
        store.ensure_parent_path_index()  # ne doit pas lever au second appel