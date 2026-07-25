#!/usr/bin/env python3
"""Script d'ingestion complete — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

#!/usr/bin/env python3
"""Script d'ingestion batch avec découverte dynamique des manifestes."""

import argparse
import sys
from pathlib import Path

import structlog

# Ajouter src/ au path si exécuté depuis la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings, get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.ingestion.pipeline import run_ingestion
from qdrant_client import QdrantClient

logger = structlog.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TRADE Ingestion Pipeline")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="Répertoire des manifestes YAML")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Répertoire du corpus raw/")
    parser.add_argument("--source-id", type=str, default=None, help="Ingestion d'une seule source")
    parser.add_argument("--qdrant-url", type=str, default=None, help="URL Qdrant (override)")
    args = parser.parse_args()

    settings = get_settings()

    # Override si fourni en CLI
    qdrant_url = args.qdrant_url or f"http://{settings.qdrant_host}:{settings.qdrant_port}"
    client = QdrantClient(qdrant_url)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    store.ensure_collection()

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        dtype=settings.embedding_dtype,
        backend=settings.embedding_backend,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    report = run_ingestion(
        manifest_dir=args.manifest_dir,
        raw_dir=args.raw_dir,
        store=store,
        embedder=embedder,
        source_id_filter=args.source_id,
    )

    print("\n=== Ingestion Report ===")
    print(f"Documents: {report.total_documents}")
    print(f"Chunks: {report.total_chunks}")
    print(f"Errors: {len(report.errors)}")
    for source_id, src in report.by_source.items():
        print(f"  {source_id}: {src['documents']} docs, {src['chunks']} chunks, {len(src['errors'])} errors")
    if report.errors:
        print("\nGlobal errors:")
        for err in report.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()