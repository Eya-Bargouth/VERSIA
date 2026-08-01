"""Test d'intégration sur le corpus réel (marqué slow)."""

import pytest
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.ingestion.pipeline import run_ingestion

pytestmark = [pytest.mark.phase2, pytest.mark.slow]


class TestRealPipeline:
    def test_ingestion_real_corpus(self, project_root):
        """Test complet du pipeline sur les vrais documents."""
        manifest_dir = project_root / "tests" / "fixtures" / "manifests"
        raw_dir = project_root / "raw"

        client = QdrantClient(":memory:")
        store = QdrantStore(client=client, collection_name="real_corpus")
        store.ensure_collection()

        settings = get_settings()

        import os
        
        # Le prompt demande l'utilisation du vrai modèle BGE-M3
        embedder = BGEEmbedder()

        source_id_filter = os.environ.get("TRADE_TEST_SOURCE")
        file_filter = os.environ.get("TRADE_TEST_FILE")

        report = run_ingestion(
            manifest_dir=manifest_dir,
            raw_dir=raw_dir,
            store=store,
            embedder=embedder,
            source_id_filter=source_id_filter,
            file_filter=file_filter,
        )

        assert report.total_documents > 0
        assert report.total_chunks > 0

        # Tolérer quelques erreurs sur les specs complexes ou parser manquants,
        # mais vérifier que chaque source a au moins un document parsé.
        # NB: en run filtré (TRADE_TEST_SOURCE/TRADE_TEST_FILE), certaines
        # sources sont absentes de report.by_source par construction ;
        # on ne vérifie donc que celles réellement présentes dans ce run.
        if "owasp_cheatsheets" in report.by_source:
            assert report.by_source["owasp_cheatsheets"]["documents"] > 0

        if "stripe_specs" in report.by_source:
            assert report.by_source["stripe_specs"]["documents"] > 0

        if "binance_spot" in report.by_source:
            assert report.by_source["binance_spot"]["documents"] > 0

        if "alpaca_incidents" in report.by_source:
            assert report.by_source["alpaca_incidents"]["documents"] > 0

        # Le parser PDF Docling n'est pas implémenté (placeholder),
        # Donc gdpr_regulation générera probablement une erreur "No builder supports".
        # On ne vérifie sa présence que si aucun filtre de source n'est actif
        # (sinon gdpr_regulation est légitimement absent du rapport).
        if source_id_filter is None:
            assert "gdpr_regulation" in report.by_source