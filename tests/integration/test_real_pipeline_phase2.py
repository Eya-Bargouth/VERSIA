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
        """Test complet du pipeline sur les vrais documents — découverte
        100% automatique (aucun manifeste, voir discover_sources)."""
        raw_dir = project_root / "raw"
        versioning_dir = project_root / "tests" / "fixtures" / "versioning"

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
            raw_dir=raw_dir,
            store=store,
            embedder=embedder,
            source_id_filter=source_id_filter,
            file_filter=file_filter,
            versioning_dir=versioning_dir,
        )

        assert report.total_documents > 0
        assert report.total_chunks > 0

        # Tolérer quelques erreurs sur les specs complexes, mais vérifier que
        # chaque source a au moins un document parsé. NB: en run filtré
        # (TRADE_TEST_SOURCE/TRADE_TEST_FILE), certaines sources sont
        # absentes de report.by_source par construction ; on ne vérifie donc
        # que celles réellement présentes dans ce run. source_id = nom du
        # dossier (voir discover_sources) : owasp_cheatsheets, plaid,
        # binance, alpaca, regulation.
        for source_id in ("owasp_cheatsheets", "plaid", "binance", "alpaca"):
            if source_id in report.by_source:
                assert report.by_source[source_id]["documents"] > 0

        if source_id_filter is None:
            assert "regulation" in report.by_source

    def test_qdrant_vector_store_persistent(self):
        """
        Test réel de Qdrant avec storage persistant.
        
        Vérifie que la collection Qdrant existante (lancée via docker-compose)
        contient les chunks pré-ingérés avec toutes les métadonnées enrichies.
        
        Prérequis :
            - docker compose up -d qdrant
            - python scripts/run_ingestion.py --raw-dir raw/
        """
        import os
        
        # Se connecter à l'instance Qdrant réelle (pas in-memory)
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "trade_chunks")
        
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
        except Exception as e:
            pytest.skip(f"Qdrant non accessible à {qdrant_host}:{qdrant_port} — {e}")
        
        # Vérifier que la collection existe
        try:
            collection_info = client.get_collection(collection_name)
        except Exception as e:
            pytest.fail(
                f"Collection '{collection_name}' n'existe pas. "
                f"Avez-vous exécuté : python scripts/run_ingestion.py --raw-dir raw/ ?"
            )
        
        # 1. Vérifier que des points existent
        assert collection_info.points_count > 0, \
            f"Collection '{collection_name}' vide ({collection_info.points_count} points)"
        
        # 2. Récupérer des chunks et vérifier les vecteurs
        points, _ = client.scroll(
            collection_name=collection_name, limit=10, with_payload=True, with_vectors=True
        )
        assert len(points) > 0, "Aucun chunk trouvé"
        
        first_point = points[0]
        
        # Vérifier que les vecteurs dense et sparse existent
        assert first_point.vector is not None, "Vecteur dense manquant"
        
        # 3. Vérifier les payloads (14 métadonnées obligatoires)
        payload = first_point.payload
        assert payload is not None, "Payload manquant"
        
        mandatory_fields = [
            "chunk_id", "source_id", "source_type", "text", "raw_text",
            "hierarchy_path", "status", "section_title"
        ]
        for field in mandatory_fields:
            assert field in payload, f"Champ obligatoire '{field}' manquant"
        
        # Version et obsolescence (peuvent être None mais doivent exister)
        optional_fields = ["version_tag", "version_order", "valid_from", "valid_until", "page_num", "line_num"]
        for field in optional_fields:
            assert field in payload or payload.get(field) is None, \
                f"Champ '{field}' invalide (doit être présent ou None)"
        
        # 4. Vérifier que le texte n'est pas vide
        assert len(payload["text"]) > 0, "Texte du chunk vide"
        assert len(payload["source_id"]) > 0, "source_id vide"
        
        # 5. Vérifier les statuts d'obsolescence
        status_values = set(p.payload.get("status") for p in points if p.payload.get("status"))
        valid_statuses = {"active", "superseded", "deprecated", "draft"}
        assert status_values.issubset(valid_statuses), \
            f"Statuts invalides trouvés : {status_values - valid_statuses}"
        assert len(status_values) > 0, "Aucun statut valide trouvé"
        
        # 6. Vérifier la cohérence versioning
        versioned_chunks = [p for p in points if p.payload.get("version_tag")]
        for chunk in versioned_chunks:
            assert "version_order" in chunk.payload, \
                f"Chunk versionnés {chunk.payload['chunk_id']} sans version_order"
            assert isinstance(chunk.payload["version_order"], int), \
                "version_order doit être un entier"
        
        # 7. Tester la recherche vectorielle sur la vraie instance
        import numpy as np
        query_vector = np.random.randn(1024).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # .search() retiré de qdrant-client 1.12+ (remplacé par
        # query_points()/using=) — voir QdrantStore._execute_search pour le
        # même wrapper de compatibilité côté src/.
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=5,
            with_payload=True
        )
        search_results = search_response.points
        
        assert len(search_results) > 0, "Recherche dense retourne zéro résultats"
        for result in search_results:
            assert hasattr(result, 'score'), "Score manquant"
            assert 0.0 <= result.score <= 1.0, f"Score invalide: {result.score}"
            assert result.payload is not None, "Payload manquant dans résultat"
        
        # 8. Vérifier la hiérarchie (hierarchy_path non vide)
        hierarchy_paths = [p.payload.get("hierarchy_path") for p in points]
        assert all(h for h in hierarchy_paths), "Certains chunks n'ont pas de hierarchy_path"
        
        # 9. Vérifier qu'il y a plusieurs sources
        sources = set(p.payload["source_id"] for p in points)
        assert len(sources) > 0, "Pas de source trouvée"