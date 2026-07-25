#!/usr/bin/env python3
"""Mesure provisoire Baseline A (dense-only) et Baseline B (dense+sparse+RRF minimal)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from qdrant_client import QdrantClient

logger = structlog.get_logger(__name__)

# Requêtes de test artisanales (5-10) — pas le jeu de test officiel (Phase 5)
TEST_QUERIES = [
    "What parameter is required for POST /v1/orders?",
    "How to store passwords securely?",
    "List all orders endpoint",
    "API latency spike incident",
    "Authentication cheat sheet",
]


def rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """RRF client-side minimal."""
    scores: dict[str, float] = {}
    ranks_dense = {str(r.chunk_id): idx + 1 for idx, r in enumerate(dense_results)}
    ranks_sparse = {str(r.chunk_id): idx + 1 for idx, r in enumerate(sparse_results)}

    all_ids = set(ranks_dense.keys()) | set(ranks_sparse.keys())
    for cid in all_ids:
        score = 0.0
        if cid in ranks_dense:
            score += 1.0 / (k + ranks_dense[cid])
        if cid in ranks_sparse:
            score += 1.0 / (k + ranks_sparse[cid])
        scores[cid] = score

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids[:10]


def main():
    parser = argparse.ArgumentParser(description="TRADE Baseline Measurement")
    parser.add_argument("--qdrant-url", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    qdrant_url = args.qdrant_url or f"http://{settings.qdrant_host}:{settings.qdrant_port}"
    client = QdrantClient(qdrant_url)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    print("\n=== Baseline A : Dense-only ===")
    for q in TEST_QUERIES:
        dense_vec, _ = embedder.embed_query(q)
        results = store.search_dense(dense_vec, k=5)
        print(f"  Query: {q[:50]}... -> {len(results)} results")

    print("\n=== Baseline B : Dense + Sparse + RRF ===")
    for q in TEST_QUERIES:
        dense_vec, sparse_vec = embedder.embed_query(q)
        dense_res = store.search_dense(dense_vec, k=10)
        sparse_res = store.search_sparse(sparse_vec, k=10)
        fused = rrf_fusion(dense_res, sparse_res)
        print(f"  Query: {q[:50]}... -> {len(fused)} fused results")

    print("\nNOTE: Ceci est une mesure provisoire. L'évaluation formelle (50 questions + RAGAS) est un livrable de la Phase 5.")


if __name__ == "__main__":
    main()