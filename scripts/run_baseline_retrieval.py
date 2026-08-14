#!/usr/bin/env python3
"""Phase 5 — Baselines A/B/C (retrieval seul, spec §9) :

  A. Dense-only (BGE-M3, DOM hiérarchique, Contextual Retrieval)
  B. A + Sparse (SPLADE) + RRF client-side
  C. B + Reranker (BGE-Reranker-v2-m3)

Métriques : Recall@k, Precision@k, MRR, nDCG (src/evaluation/retrieval_metrics.py),
sur le jeu de questions annoté (data/eval/questions_v1.jsonl si présent, sinon
le sous-ensemble déjà généré par template — factuelles + conflit_versions,
les deux seules catégories avec expected_sources garantis exacts).
"""

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.evaluation.retrieval_metrics import aggregate, evaluate_query
from src.retrieval.hybrid_retriever import HybridRetriever

K_VALUES = [5, 10]
TOP_K_RETRIEVE = 10

FINAL_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_v1.jsonl"
DRAFT_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_v1_draft.jsonl"
TEMPLATE_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_generated_template.jsonl"
OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "baseline_A_B_C_results.json"


def load_questions() -> list[dict]:
    path = next((p for p in (FINAL_QUESTIONS, DRAFT_QUESTIONS, TEMPLATE_QUESTIONS) if p.exists()), TEMPLATE_QUESTIONS)
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Seules les questions avec expected_sources non vide sont évaluables en retrieval pur.
    questions = [q for q in questions if q.get("expected_sources")]
    print(f"Chargé {len(questions)} questions depuis {path.name}")
    return questions


def run_baseline(name: str, questions: list[dict], run_fn) -> dict:
    per_query = []
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        results = run_fn(q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        per_query.append(evaluate_query(results, q["expected_sources"], K_VALUES))

    agg = aggregate(per_query)
    agg["latency_ms_avg"] = sum(latencies) / len(latencies) if latencies else 0.0
    agg["latency_ms_p95"] = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
    print(f"\n=== Baseline {name} ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return agg


def main():
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        dtype=settings.embedding_dtype,
        backend=settings.embedding_backend,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    questions = load_questions()
    if not questions:
        print("Aucune question évaluable trouvée — abandon.")
        return

    retriever_no_rerank = HybridRetriever(store=store, embedder=embedder, use_reranker=False)
    retriever_with_rerank = HybridRetriever(store=store, embedder=embedder, use_reranker=True)
    retriever_with_rerank.warm_up()  # charge le modèle avant la boucle chronométrée, pas pendant

    def to_result_list(retrieve_output: dict) -> list[dict]:
        return retrieve_output.get("results", [])

    # Baseline A — dense-only : on force le repli dense en passant k_sparse=0
    # n'est pas supporté nativement, donc on appelle DenseSearch directement.
    def run_a(query: str):
        dense_vec, _ = embedder.embed_query(query)
        raw = retriever_no_rerank.dense_search.search_dense(dense_vec, k=TOP_K_RETRIEVE)
        return [{"payload": r.payload, "score": r.score, "chunk_id": str(r.chunk_id)} for r in raw]

    # Baseline B — hybride dense+sparse+RRF, pas de reranker.
    def run_b(query: str):
        out = retriever_no_rerank.retrieve(query=query, top_k=TOP_K_RETRIEVE, k_dense=20, k_sparse=20)
        return to_result_list(out)

    # Baseline C — hybride + reranker.
    def run_c(query: str):
        out = retriever_with_rerank.retrieve(query=query, top_k=TOP_K_RETRIEVE, k_dense=20, k_sparse=20)
        return to_result_list(out)

    results = {
        "A_dense_only": run_baseline("A — Dense-only", questions, run_a),
        "B_hybrid_rrf": run_baseline("B — Hybrid + RRF", questions, run_b),
        "C_hybrid_rrf_reranker": run_baseline("C — Hybrid + RRF + Reranker", questions, run_c),
    }
    results["_meta"] = {"n_questions": len(questions), "k_values": K_VALUES, "top_k_retrieve": TOP_K_RETRIEVE}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats écrits dans {OUT_PATH}")


if __name__ == "__main__":
    main()
