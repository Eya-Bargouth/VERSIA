#!/usr/bin/env python3
"""Diagnostic (pas un fix) — Sufficiency Precision basse (0.625, cible >0.85,
voir docs/evaluation_report.md §9) : cause suspectée, jamais vérifiée, que le
retrieval ne ramène pas toutes les sources attendues dans le top-k pour les
questions multi-sources, avant même que SufficiencyChecker ne juge quoi que
ce soit. Ce script mesure cette hypothèse sur le vrai pipeline (Qdrant +
BGE-M3 réels) sans rien corriger : pour chaque question multi-sources du jeu
de test, compare expected_sources aux source_id réellement présents dans le
top-k retrieval de production (mêmes k_dense/k_sparse/top_k que QueryPipeline).

Usage : python scripts/diagnose_multisource_retrieval.py
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.evaluation.retrieval_metrics import is_relevant
from src.retrieval.hybrid_retriever import HybridRetriever

QUESTIONS_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_v1.jsonl"
TOP_K = 10  # même défaut que QueryPipeline.answer()


def main():
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    embedder = BGEEmbedder(
        model_name=settings.embedding_model, dtype=settings.embedding_dtype,
        backend=settings.embedding_backend, device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    retriever = HybridRetriever(store=store, embedder=embedder, use_reranker=True)
    retriever.warm_up()

    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    multi = [q for q in questions if len(q.get("expected_sources") or []) > 1]
    print(f"{len(multi)} questions multi-sources sur {len(questions)}.\n")

    n_all_found, n_partial, n_none = 0, 0, 0
    for q in multi:
        out = retriever.retrieve(
            query=q["question"], top_k=TOP_K,
            k_dense=settings.retrieval_k_dense, k_sparse=settings.retrieval_k_sparse,
        )
        chunk_payloads = [c.get("payload", {}) for c in out["results"]]
        expected = set(q["expected_sources"])
        found = {exp for exp in expected if any(is_relevant(p, [exp]) for p in chunk_payloads)}
        missing = expected - found

        if not missing:
            n_all_found += 1
            status = "OK"
        elif found:
            n_partial += 1
            status = "PARTIEL"
        else:
            n_none += 1
            status = "AUCUNE"

        print(f"[{status}] {q['question'][:70]}")
        print(f"    attendu={sorted(expected)} trouvé={sorted(found)} manquant={sorted(missing)}")

    print(f"\n=== Résumé (top_k={TOP_K}, k_dense=k_sparse={settings.retrieval_k_dense}) ===")
    print(f"Toutes sources présentes : {n_all_found}/{len(multi)}")
    print(f"Partiellement présentes  : {n_partial}/{len(multi)}")
    print(f"Aucune source attendue   : {n_none}/{len(multi)}")


if __name__ == "__main__":
    main()
