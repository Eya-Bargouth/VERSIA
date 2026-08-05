#!/usr/bin/env python3
"""Mesure provisoire Baseline A (dense-only) et Baseline B (dense+sparse+RRF minimal).

Persiste un artefact JSON horodaté dans docs/audits/baselines/ à chaque run,
pour que la mesure soit vérifiable a posteriori (et pas seulement un print()
éphémère dans un terminal).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "audits" / "baselines",
        help="Répertoire où persister l'artefact JSON horodaté.",
    )
    args = parser.parse_args()

    settings = get_settings()
    qdrant_url = args.qdrant_url or f"http://{settings.qdrant_host}:{settings.qdrant_port}"
    client = QdrantClient(qdrant_url)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    collection_info = client.get_collection(settings.qdrant_collection_name)

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    # Warm-up: the first embed_query() call lazily loads BGE-M3 into GPU memory
    # (several seconds) — exclude that one-time cost from the latency stats.
    embedder.embed_query("warm-up query")

    baseline_a_runs = []
    print("\n=== Baseline A : Dense-only ===")
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        dense_vec, _ = embedder.embed_query(q)
        results = store.search_dense(dense_vec, k=5)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  Query: {q[:50]}... -> {len(results)} results ({latency_ms:.1f}ms)")
        baseline_a_runs.append({
            "query": q,
            "results_count": len(results),
            "latency_ms": round(latency_ms, 1),
            "top1_score": results[0].score if results else None,
        })

    baseline_b_runs = []
    print("\n=== Baseline B : Dense + Sparse + RRF ===")
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        dense_vec, sparse_vec = embedder.embed_query(q)
        dense_res = store.search_dense(dense_vec, k=10)
        sparse_res = store.search_sparse(sparse_vec, k=10)
        fused = rrf_fusion(dense_res, sparse_res)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  Query: {q[:50]}... -> {len(fused)} fused results ({latency_ms:.1f}ms)")
        baseline_b_runs.append({
            "query": q,
            "fused_results_count": len(fused),
            "latency_ms": round(latency_ms, 1),
        })

    print("\nNOTE: Ceci est une mesure provisoire. L'évaluation formelle (50 questions + RAGAS) est un livrable de la Phase 5.")

    def _stats(runs: list) -> dict:
        latencies = sorted(r["latency_ms"] for r in runs)
        n = len(latencies)
        return {
            "avg_ms": round(sum(latencies) / n, 1),
            "p50_ms": latencies[n // 2],
            "p95_ms": latencies[min(n - 1, int(n * 0.95))],
        }

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": qdrant_url,
        "collection": settings.qdrant_collection_name,
        "collection_points_count": collection_info.points_count,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "note": "Mesure provisoire artisanale (5 requêtes) — pas le jeu de test officiel Phase 5.",
        "baseline_a_dense_only": {"runs": baseline_a_runs, "stats": _stats(baseline_a_runs)},
        "baseline_b_dense_sparse_rrf": {"runs": baseline_b_runs, "stats": _stats(baseline_b_runs)},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"baseline_A_B_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nArtefact persisté : {out_path}")


if __name__ == "__main__":
    main()