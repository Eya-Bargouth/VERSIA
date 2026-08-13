"""Tests découverte automatique des sources (remplace l'ancienne découverte
de manifestes, Phase 5 — plus de manifeste à écrire)."""

import pytest
from pathlib import Path

from src.ingestion.pipeline import discover_sources

pytestmark = pytest.mark.phase2


class TestSourceDiscovery:
    def test_discover_sources_from_directory_tree(self, tmp_path):
        """Chaque dossier contenant directement des fichiers supportés
        devient une source — aucune déclaration nécessaire."""
        (tmp_path / "stripe").mkdir()
        (tmp_path / "stripe" / "spec.yaml").write_text("openapi: '3.0.3'\npaths: {}\n", encoding="utf-8")
        (tmp_path / "owasp-cheatsheets").mkdir()
        (tmp_path / "owasp-cheatsheets" / "auth.md").write_text("# Auth\n", encoding="utf-8")

        sources = discover_sources(tmp_path)
        source_ids = {s.source_id for s in sources}
        assert "stripe" in source_ids
        # Le tiret est assaini en underscore (source_id ~ ^[a-zA-Z0-9_]+$).
        assert "owasp_cheatsheets" in source_ids

    def test_nested_directories_each_become_their_own_source(self, tmp_path):
        """Peu importe la profondeur : specs-api/stripe/ ET specs-api/binance/
        sont deux sources distinctes, "specs-api" lui-même n'en est pas une
        (il ne contient aucun fichier directement, seulement des sous-dossiers)."""
        (tmp_path / "specs-api" / "stripe").mkdir(parents=True)
        (tmp_path / "specs-api" / "stripe" / "a.yaml").write_text("a: 1\n", encoding="utf-8")
        (tmp_path / "specs-api" / "binance").mkdir(parents=True)
        (tmp_path / "specs-api" / "binance" / "b.yaml").write_text("b: 1\n", encoding="utf-8")

        sources = discover_sources(tmp_path)
        source_ids = {s.source_id for s in sources}
        assert source_ids == {"stripe", "binance"}

    def test_directory_with_no_supported_files_is_not_a_source(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        (tmp_path / "empty_dir" / "readme.txt").write_text("hello", encoding="utf-8")

        sources = discover_sources(tmp_path)
        assert sources == []

    def test_versioning_override_loaded_when_present(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "stripe").mkdir(parents=True)
        (raw_dir / "stripe" / "spec3-v1.yaml").write_text("a: 1\n", encoding="utf-8")

        versioning_dir = tmp_path / "versioning"
        versioning_dir.mkdir()
        (versioning_dir / "stripe.yaml").write_text(
            "pattern: 'spec3-(?P<version>.+)\\.yaml'\norder: [v1, v2]\n", encoding="utf-8"
        )

        sources = discover_sources(raw_dir, versioning_dir=versioning_dir)
        stripe = next(s for s in sources if s.source_id == "stripe")
        assert stripe.version_pattern == r"spec3-(?P<version>.+)\.yaml"
        assert stripe.version_order == ["v1", "v2"]

    def test_source_without_versioning_override_stays_unversioned(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "gdpr").mkdir(parents=True)
        (raw_dir / "gdpr" / "reg.md").write_text("# GDPR\n", encoding="utf-8")

        sources = discover_sources(raw_dir, versioning_dir=tmp_path / "no_such_dir")
        gdpr = next(s for s in sources if s.source_id == "gdpr")
        assert gdpr.version_pattern is None
        assert gdpr.version_order is None

    def test_real_raw_dir_and_versioning_dir_discover_five_sources(self, project_root):
        """Contre le vrai corpus du projet : 5 sources attendues, seule
        "stripe" versionnée (voir tests/fixtures/versioning/stripe.yaml)."""
        raw_dir = project_root / "raw"
        versioning_dir = project_root / "tests" / "fixtures" / "versioning"
        if not raw_dir.exists():
            pytest.skip("raw/ absent de cet environnement")

        sources = discover_sources(raw_dir, versioning_dir=versioning_dir)
        source_ids = {s.source_id for s in sources}
        assert {"stripe", "binance", "owasp_cheatsheets", "regulation", "alpaca"} <= source_ids

        stripe = next(s for s in sources if s.source_id == "stripe")
        assert stripe.version_pattern is not None
        assert stripe.version_order == ["legacy", "v2213", "v2293", "v2323"]
