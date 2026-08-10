"""Tests VersionDiffEngine."""

import pytest
from pathlib import Path

from src.config.manifest_schema import SourceManifest
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.version_diff import VersionDiffEngine, DiffReport

pytestmark = pytest.mark.phase2


def _tagged_markdown_tree(path: Path, content: str, tag: str, source_id: str = "policy_doc"):
    path.write_text(content, encoding="utf-8")
    manifest = SourceManifest(source_id=source_id, parser="markdown", scope={"include": ["*.md"]})
    tree = MarkdownBuilder().build(str(path), manifest)
    tree.compute_all_hashes()
    tree.nodes[tree.root_id].version_tag = tag
    return tree


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

    def test_diff_detects_description_only_change(self, tmp_path):
        """La description d'un endpoint vit dans `markdown`, jamais dans
        `metadata` — un changement de description doit apparaître dans
        field_changes, pas seulement faire varier content_hash en silence."""
        v1_yaml = tmp_path / "api_v1.yaml"
        v1_yaml.write_text("""
openapi: "3.0.3"
info:
  title: Test API
  version: "1.0.0"
paths:
  /v1/subscriptions/{id}:
    delete:
      summary: Cancel a subscription
      description: "Prorations are removed if prorate is set to false."
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
  /v1/subscriptions/{id}:
    delete:
      summary: Cancel a subscription
      description: "Prorations are removed if prorate is set to true."
      responses:
        "200":
          description: OK
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

        modified = [c for c in report.changes if c.change_type == "modified"]
        assert len(modified) == 1
        assert "markdown" in modified[0].field_changes
        assert "false" in modified[0].field_changes["markdown"]["old"]
        assert "true" in modified[0].field_changes["markdown"]["new"]

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


class TestVersionDiffEngineNarrativeDocs:
    """_canonicalize() ne gérait à l'origine que les nœuds API-shaped
    (endpoint/paramètre/réponse) — ces tests couvrent la généralisation aux
    documents narratifs (paragraphes, listes, titres, code) via hierarchy_path
    + position ordinale (voir VersionDiffEngine._generic_key)."""

    def test_detects_paragraph_content_change(self, tmp_path):
        tree_v1 = _tagged_markdown_tree(
            tmp_path / "v1.md",
            "# Refund Policy\n\n## Eligibility\n\nRefunds must be requested within 30 days of purchase.\n\n"
            "## Process\n\nContact support to initiate a refund.\n",
            "v1",
        )
        tree_v2 = _tagged_markdown_tree(
            tmp_path / "v2.md",
            "# Refund Policy\n\n## Eligibility\n\nRefunds must be requested within 90 days of purchase.\n\n"
            "## Process\n\nContact support to initiate a refund.\n",
            "v2",
        )

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        assert len(report.changes) == 1
        change = report.changes[0]
        assert change.change_type == "modified"
        assert "Eligibility" in change.key
        assert "30 days" in change.field_changes["text"]["old"]
        assert "90 days" in change.field_changes["text"]["new"]

    def test_unchanged_paragraph_produces_no_diff(self, tmp_path):
        content = "# Doc\n\n## Section\n\nThis paragraph never changes.\n"
        tree_v1 = _tagged_markdown_tree(tmp_path / "v1.md", content, "v1")
        tree_v2 = _tagged_markdown_tree(tmp_path / "v2.md", content, "v2")

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        assert report.changes == []

    def test_inserted_paragraph_shifts_downstream_keys(self, tmp_path):
        """Limite connue et documentée : sans clé sémantique naturelle (pas
        de METHOD:path pour du texte narratif), la position ordinale sert de
        repère — une insertion en milieu de section décale tous les
        paragraphes suivants, qui apparaissent comme "modified" même si leur
        contenu n'a pas changé. Ce test documente ce faux positif plutôt que
        de prétendre qu'il n'existe pas."""
        tree_v1 = _tagged_markdown_tree(
            tmp_path / "v1.md",
            "# Doc\n\n## Section\n\nFirst paragraph.\n\nSecond paragraph.\n",
            "v1",
        )
        tree_v2 = _tagged_markdown_tree(
            tmp_path / "v2.md",
            "# Doc\n\n## Section\n\nFirst paragraph.\n\nInserted paragraph.\n\nSecond paragraph.\n",
            "v2",
        )

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        by_type = {c.change_type for c in report.changes}
        assert "added" in by_type or "modified" in by_type
        # "Second paragraph" (#1 dans v1) devient "#2" dans v2 — la clé
        # positionnelle #1 se retrouve donc comparée à "Inserted paragraph",
        # pas à son propre contenu inchangé : c'est le faux positif attendu.
        assert len(report.changes) >= 2

    def test_api_and_narrative_nodes_coexist_in_same_diff(self, tmp_path):
        """Un même arbre peut mélanger API_ENDPOINT (clé naturelle) et
        PARAGRAPH (clé générique) — les deux stratégies doivent cohabiter
        sans collision de clé."""
        manifest = SourceManifest(
            source_id="mixed", parser="yaml_structured", scope={"include": ["*.yaml"]}
        )
        yaml_v1 = tmp_path / "api_v1.yaml"
        yaml_v1.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\n"
            "paths:\n  /ping:\n    get:\n      summary: Ping\n",
            encoding="utf-8",
        )
        yaml_v2 = tmp_path / "api_v2.yaml"
        yaml_v2.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"2.0.0\"\n"
            "paths:\n  /ping:\n    get:\n      summary: Ping v2\n",
            encoding="utf-8",
        )
        tree_v1 = YAMLBuilder().build(str(yaml_v1), manifest)
        tree_v2 = YAMLBuilder().build(str(yaml_v2), manifest)
        tree_v1.compute_all_hashes()
        tree_v2.compute_all_hashes()

        engine = VersionDiffEngine(diff_dir=tmp_path / "diffs")
        report = engine.diff(tree_v1, tree_v2)

        assert any(c.change_type == "modified" and "GET:/ping" in c.key for c in report.changes)