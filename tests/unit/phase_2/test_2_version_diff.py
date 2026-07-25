"""Tests VersionDiffEngine."""

import pytest
from pathlib import Path

from src.config.manifest_schema import SourceManifest
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.version_diff import VersionDiffEngine, DiffReport

pytestmark = pytest.mark.phase2


class TestVersionDiffEngine:
    def test_diff_detects_added_removed_modified(self, tmp_path):
        # Créer deux versions d'un OpenAPI minimal
        v1_yaml = tmp_path / "api_v1.yaml"
        v1_yaml.write_text("""
openapi: "3.0.3"
info:
  title: Test API
  version: "1.0.0"
paths:
  /v1/orders:
    post:
      summary: Create order
      parameters:
        - name: symbol
          in: query
          required: true
          schema:
            type: string
      responses:
        "200":
          description: OK
""", encoding="utf-8")

        v2_yaml = tmp_path / "api_v2.yaml"
        v2_yaml.write_text("""
openapi: "3.0.3"
info:
  title: Test API
  version: "2.0.0"
paths:
  /v1/orders:
    post:
      summary: Create order
      parameters:
        - name: symbol
          in: query
          required: true
          schema:
            type: string
        - name: quantity
          in: query
          required: true
          schema:
            type: number
      responses:
        "200":
          description: Order created
        "400":
          description: Bad request
""", encoding="utf-8")

        manifest = SourceManifest(
            source_id="test_api",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
        )
        tree_v1 = YAMLBuilder().build(str(v1_yaml), manifest)
        tree_v2 = YAMLBuilder().build(str(v2_yaml), manifest)

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        assert isinstance(report, DiffReport)
        assert report.source_id == "test_api"
        assert any(c.change_type == "added" for c in report.changes)
        assert any(c.change_type == "modified" for c in report.changes)

    def test_load_diff_returns_none_when_missing(self, tmp_path):
        engine = VersionDiffEngine(diff_dir=tmp_path)
        result = engine.load_diff("nonexistent", "v1", "v2")
        assert result is None

    def test_diff_serialization_roundtrip(self, tmp_path):
        v1_yaml = tmp_path / "api_v1.yaml"
        v1_yaml.write_text("""
openapi: "3.0.3"
info:
  title: X
  version: "1.0.0"
paths:
  /ping:
    get:
      summary: Ping
""", encoding="utf-8")
        v2_yaml = tmp_path / "api_v2.yaml"
        v2_yaml.write_text("""
openapi: "3.0.3"
info:
  title: X
  version: "2.0.0"
paths:
  /ping:
    get:
      summary: Ping v2
""", encoding="utf-8")

        manifest = SourceManifest(
            source_id="x",
            parser="yaml_structured",
            scope={"include": ["*.yaml"]},
        )
        tree_v1 = YAMLBuilder().build(str(v1_yaml), manifest)
        tree_v2 = YAMLBuilder().build(str(v2_yaml), manifest)

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        loaded = engine.load_diff(report.source_id, report.version_from, report.version_to)
        assert loaded is not None
        assert len(loaded.changes) == len(report.changes)