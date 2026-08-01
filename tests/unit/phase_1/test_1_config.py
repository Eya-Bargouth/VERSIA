"""Tests unitaires pour la configuration."""

import pytest

from src.config.manifest_schema import ChunkingPolicy, SourceManifest
from src.config.settings import Settings


pytestmark = pytest.mark.phase1


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.llm_provider == "ollama"
        assert s.llm_model == "qwen3:4b"

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HOST", "qdrant.test")
        monkeypatch.setenv("LLM_MODEL", "llama3:8b")
        s = Settings()
        assert s.qdrant_host == "qdrant.test"
        assert s.llm_model == "llama3:8b"


class TestManifestSchema:
    def test_source_manifest_creation(self):
        m = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
        )
        assert m.source_id == "test_api"
        assert m.parser == "yaml_structured"
        assert m.chunking_policy.semantic_unit == "paragraph"

    def test_manifest_scope_must_have_include(self):
        with pytest.raises(ValueError, match="include"):
            SourceManifest(
                source_id="bad",
                parser="markdown",
                scope={"exclude": ["*"]},
            )

    def test_manifest_from_yaml(self, tmp_path):
        yaml_content = """
manifest_version: "1.0.0"
source_id: my_source
parser: markdown
scope:
  include:
    - "*.md"
chunking_policy:
  semantic_unit: paragraph
validity:
  status: active
"""
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml_content, encoding="utf-8")
        m = SourceManifest.from_yaml(path)
        assert m.source_id == "my_source"
        assert m.validity.status == "active"
