"""Tests unitaires pour les DOM builders."""

import pytest

from src.config.manifest_schema import ChunkingPolicy, SourceManifest
from src.dom.builders.json_builder import JSONBuilder
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.dom.models import NodeType

pytestmark = pytest.mark.phase1


class TestRootNodeVersioningAndValidity:
    """AbstractDOMBuilder._create_root_node doit appliquer
    manifest.versioning/manifest.validity à la racine — gap Phase 2 corrigé
    en Phase 4 (voir docs/PHASE_4_SUMMARY.md)."""

    def test_filename_pattern_sets_version_tag_and_order(self, tmp_path):
        v_path = tmp_path / "spec3-v2213.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\npaths: {}\n",
            encoding="utf-8",
        )
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            versioning={
                "strategy": "filename_pattern",
                "pattern": r"spec3-(?P<version>.+)\.yaml",
                "order": ["legacy", "v2213", "v2293"],
            },
        )
        tree = YAMLBuilder().build(str(v_path), manifest)
        root = tree.nodes[tree.root_id]
        assert root.version_tag == "v2213"
        assert root.version_order == 1

    def test_validity_propagated_to_root(self, tmp_path):
        v_path = tmp_path / "api.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\npaths: {}\n",
            encoding="utf-8",
        )
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            validity={"status": "deprecated", "valid_from": "2020-01-01"},
        )
        tree = YAMLBuilder().build(str(v_path), manifest)
        root = tree.nodes[tree.root_id]
        assert root.status == "deprecated"
        assert str(root.valid_from) == "2020-01-01"

    def test_no_versioning_strategy_leaves_root_untagged(self, sample_openapi_path):
        manifest = SourceManifest(
            source_id="test_api", parser="yaml_structured", scope={"include": ["*.yaml"]}
        )
        tree = YAMLBuilder().build(str(sample_openapi_path), manifest)
        root = tree.nodes[tree.root_id]
        assert root.version_tag is None
        assert root.version_order is None

    def test_pattern_not_matching_filename_does_not_crash(self, tmp_path):
        v_path = tmp_path / "unrelated_name.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\npaths: {}\n",
            encoding="utf-8",
        )
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            versioning={
                "strategy": "filename_pattern",
                "pattern": r"spec3-(?P<version>.+)\.yaml",
                "order": ["v1"],
            },
        )
        tree = YAMLBuilder().build(str(v_path), manifest)
        assert tree.nodes[tree.root_id].version_tag is None


class TestYAMLBuilder:
    def test_supports_yaml(self):
        builder = YAMLBuilder()
        assert builder.supports("test.yaml") is True
        assert builder.supports("test.yml") is True
        assert builder.supports("test.pdf") is False

    def test_build_openapi(self, sample_openapi_path):
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            chunking_policy=ChunkingPolicy(semantic_unit="api_endpoint"),
        )
        builder = YAMLBuilder()
        tree = builder.build(str(sample_openapi_path), manifest)

        assert tree.root_id in tree.nodes
        endpoints = tree.get_nodes_by_type(NodeType.API_ENDPOINT)
        assert len(endpoints) >= 2

        post_orders = [e for e in endpoints if "POST" in (e.text or "") and "/v1/orders" in (e.text or "")]
        assert len(post_orders) == 1
        assert post_orders[0].type == NodeType.API_ENDPOINT

        params = tree.get_nodes_by_type(NodeType.API_PARAMETER)
        assert len(params) >= 3

    def test_endpoint_has_markdown(self, sample_openapi_path):
        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
            chunking_policy=ChunkingPolicy(semantic_unit="api_endpoint"),
        )
        tree = YAMLBuilder().build(str(sample_openapi_path), manifest)
        endpoints = tree.get_nodes_by_type(NodeType.API_ENDPOINT)
        assert all(e.markdown is not None and len(e.markdown) > 0 for e in endpoints)


class TestMarkdownBuilder:
    def test_supports_md(self):
        builder = MarkdownBuilder()
        assert builder.supports("test.md") is True
        assert builder.supports("test.markdown") is True
        assert builder.supports("test.yaml") is False

    def test_build_markdown(self, sample_markdown_path):
        manifest = SourceManifest(
            source_id="test_md",
            parser="markdown",
            scope={"include": ["*.md"]},
            chunking_policy=ChunkingPolicy(semantic_unit="paragraph"),
        )
        builder = MarkdownBuilder()
        tree = builder.build(str(sample_markdown_path), manifest)

        headings = tree.get_nodes_by_type(NodeType.HEADING)
        assert len(headings) >= 4

        paragraphs = tree.get_nodes_by_type(NodeType.PARAGRAPH)
        assert len(paragraphs) >= 1

        code_blocks = tree.get_nodes_by_type(NodeType.CODE_BLOCK)
        assert len(code_blocks) >= 1

        list_items = tree.get_nodes_by_type(NodeType.LIST_ITEM)
        assert len(list_items) >= 6

    def test_hierarchy_integrity(self, sample_markdown_path):
        manifest = SourceManifest(
            source_id="test_md",
            parser="markdown",
            scope={"include": ["*.md"]},
        )
        tree = MarkdownBuilder().build(str(sample_markdown_path), manifest)
        errors = tree.validate_integrity()
        assert len(errors) == 0, f"Integrity errors: {errors}"


class TestJSONBuilder:
    def test_supports_json(self):
        builder = JSONBuilder()
        assert builder.supports("test.json") is True
        assert builder.supports("test.yaml") is False

    def test_build_incidents(self, sample_json_path):
        manifest = SourceManifest(
            source_id="test_incidents",
            parser="json",
            scope={"include": ["*.json"]},
            chunking_policy=ChunkingPolicy(semantic_unit="document", group_nested=["updates", "timeline"]),
        )
        builder = JSONBuilder()
        tree = builder.build(str(sample_json_path), manifest)

        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) >= 2

        root = tree.get_node(tree.root_id)
        assert root is not None
        assert root.type == NodeType.DOCUMENT

    def test_integrity(self, sample_json_path):
        manifest = SourceManifest(
            source_id="test_incidents",
            parser="json",
            scope={"include": ["*.json"]},
        )
        tree = JSONBuilder().build(str(sample_json_path), manifest)
        assert len(tree.validate_integrity()) == 0