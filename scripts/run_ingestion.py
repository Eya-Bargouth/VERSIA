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
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig
from src.llm.providers import ollama_client, vllm_client  # noqa: F401 — self-register with LLMFactory
from qdrant_client import QdrantClient

logger = structlog.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TRADE Ingestion Pipeline")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="Répertoire des manifestes YAML")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Répertoire du corpus raw/")
    parser.add_argument("--source-id", type=str, default=None, help="Ingestion d'une seule source")
    parser.add_argument("--qdrant-url", type=str, default=None, help="URL Qdrant (override)")
    parser.add_argument(
        "--prefix-method",
        type=str,
        choices=["deterministic", "llm"],
        default="deterministic",
        help=(
            "Méthode de génération du préfixe contextuel (Contextual Retrieval). "
            "'deterministic' (défaut, rapide, reproductible, sans coût GPU/LLM) "
            "ou 'llm' (résumé via le provider configuré dans .env, avec repli "
            "automatique sur 'deterministic' en cas d'échec)."
        ),
    )
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

    llm_client = None
    llm_config = None
    if args.prefix_method == "llm":
        llm_config = LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )
        llm_client = LLMFactory.create(llm_config)

    report = run_ingestion(
        manifest_dir=args.manifest_dir,
        raw_dir=args.raw_dir,
        store=store,
        embedder=embedder,
        source_id_filter=args.source_id,
        prefix_method=args.prefix_method,
        llm_client=llm_client,
        llm_config=llm_config,
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