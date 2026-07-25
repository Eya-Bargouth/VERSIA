"""Tests découverte dynamique des manifestes."""

import pytest
from pathlib import Path

from src.config.manifest_schema import SourceManifest
from src.ingestion.pipeline import discover_manifests

pytestmark = pytest.mark.phase2


class TestManifestDiscovery:
    def test_discover_all_valid_manifests(self, fixtures_dir):
        manifest_dir = fixtures_dir / "manifests"
        manifests = discover_manifests(manifest_dir)
        source_ids = {m.source_id for m in manifests}
        assert "stripe_specs" in source_ids
        assert "owasp_cheatsheets" in source_ids
        assert "binance_spot" in source_ids
        assert "gdpr_regulation" in source_ids
        assert "alpaca_incidents" in source_ids

    def test_invalid_manifest_does_not_block_others(self, tmp_path):
        # Créer un manifeste valide et un invalide
        valid = tmp_path / "valid.yaml"
        valid.write_text("""
manifest_version: \"1.0.0\"
source_id: valid_test
parser: markdown
scope:
  include:
    - \"*.md\"
""", encoding="utf-8")
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("""
manifest_version: \"1.0.0\"
source_id: bad
parser: unknown_parser
scope:
  include:
    - \"*.md\"
""", encoding="utf-8")

        manifests = discover_manifests(tmp_path)
        assert len(manifests) == 1
        assert manifests[0].source_id == "valid_test"

    def test_manifests_are_validated(self, fixtures_dir):
        manifest_dir = fixtures_dir / "manifests"
        for path in manifest_dir.glob("*.yaml"):
            m = SourceManifest.from_yaml(path)
            assert m.source_id
            assert m.parser in ("docling", "yaml_structured", "markdown", "json")