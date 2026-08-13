"""Tests unitaires pour la configuration."""

import pytest
from pydantic import ValidationError

from src.config.settings import Settings
from src.config.source_config import SourceConfig


pytestmark = pytest.mark.phase1


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.llm_provider == "ollama"
        assert s.llm_model == "qwen2.5:3b-instruct"

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HOST", "qdrant.test")
        monkeypatch.setenv("LLM_MODEL", "llama3:8b")
        s = Settings()
        assert s.qdrant_host == "qdrant.test"
        assert s.llm_model == "llama3:8b"


class TestSourceConfig:
    """SourceConfig remplace SourceManifest (Phase 5) — plus de manifeste à
    écrire, seul le versioning reste une exception déclarable, hors de ce
    modèle (voir tests/fixtures/versioning/, chargé par
    src.ingestion.pipeline.discover_sources)."""

    def test_source_config_defaults(self):
        c = SourceConfig(source_id="test_api")
        assert c.source_id == "test_api"
        assert c.source_type == "document"
        assert c.version_pattern is None
        assert c.version_order is None

    def test_source_id_must_match_pattern(self):
        with pytest.raises(ValidationError):
            SourceConfig(source_id="bad id with spaces")

    def test_versioning_override_fields(self):
        c = SourceConfig(
            source_id="stripe",
            version_pattern=r"spec3-(?P<version>.+)\.yaml",
            version_order=["legacy", "v2213"],
        )
        assert c.version_pattern == r"spec3-(?P<version>.+)\.yaml"
        assert c.version_order == ["legacy", "v2213"]
