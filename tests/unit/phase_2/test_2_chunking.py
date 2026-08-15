"""Tests chunking hiérarchique + Contextual Retrieval."""

import pytest
from unittest.mock import create_autospec

from src.config.source_config import SourceConfig
from src.dom.builders.yaml_builder import YAMLBuilder
from src.dom.models import NodeType
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage

pytestmark = pytest.mark.phase2


class TestHierarchicalChunking:
    def test_chunk_hierarchy_integrity(self, sample_openapi_path):
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        chunker = HierarchicalChunker(prefix_method="deterministic", max_prefix_tokens=100)
        chunks = chunker.chunk(tree)

        # Chunking universel : tout nœud de contenu (type DOCUMENT ici,
        # plus d'API_ENDPOINT typé) devient un chunk feuille. Filtre sur
        # raw_text non vide pour exclure le chunk structurel de la racine
        # (même NodeType.DOCUMENT que les feuilles, mais texte vide — la
        # racine elle-même n'a jamais de contenu propre).
        leaf_chunks = [c for c in chunks if c.metadata.node_type == NodeType.DOCUMENT and c.raw_text]
        assert len(leaf_chunks) >= 1

        # Vérifier contextual_prefix non vide
        for c in leaf_chunks:
            assert c.contextual_prefix

        # Vérifier format texte (vrais sauts de ligne, pas des backslash-littéraux)
        for c in leaf_chunks:
            assert c.text == f"{c.contextual_prefix}\n\n{c.raw_text}"

    def test_content_hash_stable(self, sample_openapi_path):
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        chunker = HierarchicalChunker(prefix_method="deterministic")
        chunks1 = chunker.chunk(tree)
        chunks2 = chunker.chunk(tree)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.content_hash == c2.content_hash

    def test_chunks_inherit_version_and_status_from_tree_root(self, tmp_path):
        """Un chunk n'a pas son propre version_tag/status — hérités de la
        racine de l'arbre (un fichier = une version), pas None comme avant
        la correction de ce gap (voir docs/PHASE_4_SUMMARY.md, Tâche 15).
        status reste "active" par défaut (plus de ValidityConfig par
        source, voir SourceConfig)."""
        v_path = tmp_path / "spec3-v2293.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\n"
            "paths:\n  /ping:\n    get:\n      summary: Ping\n",
            encoding="utf-8",
        )
        config = SourceConfig(
            source_id="test_api",
            version_pattern=r"spec3-(?P<version>.+)\.yaml",
            version_order=["v2213", "v2293"],
        )
        tree = YAMLBuilder().build(str(v_path), config)
        chunker = HierarchicalChunker(prefix_method="deterministic")
        chunks = chunker.chunk(tree)

        assert chunks
        for c in chunks:
            assert c.version_tag == "v2293"
            assert c.version_order == 1
            assert c.status == "active"

    def test_mock_llm_prefix(self, sample_openapi_path):
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)

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
        chunks = chunker.chunk(tree)

        # Vérifie que complete() a bien été appelé au moins une fois
        assert client.complete.called
        assert any("Contexte API Orders" in c.contextual_prefix for c in chunks)

    def test_leaf_chunks_get_linked_to_structural_parent_chunk(self, sample_openapi_path):
        """Régression : _link_chunk_hierarchy comparait chunk.node_ids[0]
        (str, ex. "9ce0eb5-...") directement contre tree.nodes (indexé par
        UUID) sans reconversion — .get() ne trouvait jamais rien, si bien
        qu'AUCUN chunk n'obtenait jamais de parent_chunk_id, sur aucune
        source, silencieusement (bug trouvé en testant l'expansion parent
        explicite via Qdrant, audit #15)."""
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        chunker = HierarchicalChunker(prefix_method="deterministic")
        chunks = chunker.chunk(tree)

        leaf_chunks = [c for c in chunks if c.metadata.node_type == NodeType.DOCUMENT and c.raw_text]
        with_parent = [c for c in leaf_chunks if c.parent_chunk_id is not None]
        assert with_parent, "aucun chunk feuille n'a de parent_chunk_id — la liaison hiérarchique est cassée"

        # Le parent référencé doit réellement exister parmi les chunks produits,
        # et lister l'enfant dans ses child_chunk_ids (lien bidirectionnel).
        chunk_by_id = {c.chunk_id: c for c in chunks}
        child = with_parent[0]
        parent = chunk_by_id[child.parent_chunk_id]
        assert child.chunk_id in parent.child_chunk_ids
