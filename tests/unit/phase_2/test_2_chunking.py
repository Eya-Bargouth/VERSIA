"""Tests chunking hiérarchique + Contextual Retrieval."""

import pytest
from unittest.mock import create_autospec

from src.config.manifest_schema import ChunkingPolicy
from src.dom.builders.yaml_builder import YAMLBuilder
from src.dom.models import NodeType
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage

pytestmark = pytest.mark.phase2


class TestHierarchicalChunking:
    def test_chunk_hierarchy_integrity(self, sample_openapi_path):
        from src.config.manifest_schema import SourceManifest
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            chunking_policy=ChunkingPolicy(semantic_unit="api_endpoint", include_parent_context=True),
        )
        tree = YAMLBuilder().build(str(sample_openapi_path), manifest)
        chunker = HierarchicalChunker(prefix_method="deterministic", max_prefix_tokens=100)
        chunks = chunker.chunk(tree, manifest.chunking_policy)

        # Vérifier qu'il y a des chunks feuilles
        leaf_chunks = [c for c in chunks if c.metadata.node_type == NodeType.API_ENDPOINT]
        assert len(leaf_chunks) >= 2

        # Vérifier contextual_prefix non vide
        for c in leaf_chunks:
            assert c.contextual_prefix
            assert "paths" in c.contextual_prefix or "POST" in c.contextual_prefix or "GET" in c.contextual_prefix

        # Vérifier format texte (vrais sauts de ligne, pas des backslash-littéraux)
        for c in leaf_chunks:
            assert c.text == f"{c.contextual_prefix}\n\n{c.raw_text}"

    def test_content_hash_stable(self, sample_openapi_path):
        from src.config.manifest_schema import SourceManifest
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            chunking_policy=ChunkingPolicy(semantic_unit="api_endpoint"),
        )
        tree = YAMLBuilder().build(str(sample_openapi_path), manifest)
        chunker = HierarchicalChunker(prefix_method="deterministic")
        chunks1 = chunker.chunk(tree, manifest.chunking_policy)
        chunks2 = chunker.chunk(tree, manifest.chunking_policy)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.content_hash == c2.content_hash

    def test_mock_llm_prefix(self, sample_openapi_path):
        from src.config.manifest_schema import SourceManifest
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            chunking_policy=ChunkingPolicy(semantic_unit="api_endpoint", include_parent_context=True),
        )
        tree = YAMLBuilder().build(str(sample_openapi_path), manifest)

        # Mock autospec : respecte BaseLLMClient, retourne toujours la même valeur
        client = create_autospec(BaseLLMClient, instance=True)
        client.complete.return_value = LLMResponse(
            content="Contexte API Orders",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="mock",
            raw_response={},
        )

        chunker = HierarchicalChunker(
            llm_client=client,
            llm_config=LLMConfig(provider="ollama", model="mistral"),
            prefix_method="llm",
            max_prefix_tokens=100,
        )
        chunks = chunker.chunk(tree, manifest.chunking_policy)
        leaf_chunks = [c for c in chunks if c.metadata.node_type == NodeType.API_ENDPOINT]

        # Vérifie que complete() a bien été appelé au moins une fois
        assert client.complete.called
        assert any("Contexte API Orders" in c.contextual_prefix for c in leaf_chunks)