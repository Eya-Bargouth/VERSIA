"""Tests VersionDiffConflictDetector — adaptation DiffReport -> ConflictReport."""

import pytest

from src.config.manifest_schema import SourceManifest
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.version_diff import VersionDiffEngine
from src.reliability.conflict.version_diff_engine import VersionDiffConflictDetector

pytestmark = pytest.mark.phase4


@pytest.fixture
def precomputed_diff(tmp_path):
    v1_yaml = tmp_path / "api_v1.yaml"
    v1_yaml.write_text(
        """
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
          required: false
          schema:
            type: string
      responses:
        "200":
          description: OK
""",
        encoding="utf-8",
    )
    v2_yaml = tmp_path / "api_v2.yaml"
    v2_yaml.write_text(
        """
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
      responses:
        "200":
          description: OK
""",
        encoding="utf-8",
    )

    manifest = SourceManifest(
        source_id="test_api", parser="yaml_structured", scope={"include": ["*.yaml"]}
    )
    tree_v1 = YAMLBuilder().build(str(v1_yaml), manifest)
    tree_v2 = YAMLBuilder().build(str(v2_yaml), manifest)
    # Aucun builder DOM ne peuple version_tag depuis le manifeste (gap
    # documenté — voir scripts/generate_version_diffs.py) : on le fixe ici
    # explicitement, comme le fait ce script pour le vrai corpus.
    tree_v1.nodes[tree_v1.root_id].version_tag = "1.0.0"
    tree_v2.nodes[tree_v2.root_id].version_tag = "2.0.0"

    diff_dir = tmp_path / "diffs"
    engine = VersionDiffEngine(diff_dir=diff_dir)
    engine.diff(tree_v1, tree_v2)
    return diff_dir


class TestVersionDiffConflictDetector:
    def test_detects_conflict_for_changed_key(self, precomputed_diff):
        detector = VersionDiffConflictDetector(diff_dir=precomputed_diff)
        report = detector.detect(
            "test_api", "1.0.0", "2.0.0", key="POST:/v1/orders|param:symbol"
        )
        assert report.conflict is True
        assert report.type == "temporal"
        assert report.method == "version_diff"
        assert report.confidence == 1.0
        assert "symbol" in report.explanation

    def test_no_conflict_for_unrelated_key(self, precomputed_diff):
        detector = VersionDiffConflictDetector(diff_dir=precomputed_diff)
        report = detector.detect("test_api", "1.0.0", "2.0.0", key="GET:/v1/unrelated")
        assert report.conflict is False

    def test_no_conflict_when_diff_not_precomputed(self, tmp_path):
        detector = VersionDiffConflictDetector(diff_dir=tmp_path / "empty")
        report = detector.detect("unknown_source", "v1", "v2")
        assert report.conflict is False
        assert report.method == "version_diff"

    def test_aggregates_all_changes_without_key(self, precomputed_diff):
        detector = VersionDiffConflictDetector(diff_dir=precomputed_diff)
        report = detector.detect("test_api", "1.0.0", "2.0.0")
        assert report.conflict is True
        assert "POST:/v1/orders|param:symbol" in report.explanation
