#!/usr/bin/env python3
"""Diagnostic (pas un fix) — Precision@10 reste sous la cible (0.655-0.668
sur B/C, cible >0.70, voir docs/evaluation_report.md) même après le
dédoublonnage inter-versions (#8/#9). Ce script identifie, pour chaque
question où Precision@10 < 1.0, la source réelle des faux positifs du
top-10 — pour savoir si le déclassement des versions restant à traiter
vient d'un autre fichier de la même source (ex. dédup incomplet), d'une
autre source entièrement, ou d'autre chose, avant de choisir une correction
(voir §"Pistes pour améliorer Precision@10").

Usage : python scripts/diagnose_precision_false_positives.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.evaluation.retrieval_metrics import is_relevant
from src.retrieval.hybrid_retriever import HybridRetriever

FINAL_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_v1.jsonl"
DRAFT_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_v1_draft.jsonl"
TEMPLATE_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "questions_generated_template.jsonl"
TOP_K = 10


def load_questions() -> list[dict]:
    path = next((p for p in (FINAL_QUESTIONS, DRAFT_QUESTIONS, TEMPLATE_QUESTIONS) if p.exists()), TEMPLATE_QUESTIONS)
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in questions if q.get("expected_sources")]


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

    questions = load_questions()
    print(f"{len(questions)} questions évaluables.\n")

    fp_kind_counter = Counter()
    n_perfect, n_with_fp = 0, 0

    for q in questions:
        out = retriever.retrieve(
            query=q["question"], top_k=TOP_K,
            k_dense=settings.retrieval_k_dense, k_sparse=settings.retrieval_k_sparse,
        )
        results = out["results"][:TOP_K]
        expected = q["expected_sources"]

        false_positives = [r for r in results if not is_relevant(r.get("payload", {}), expected)]
        if not false_positives:
            n_perfect += 1
            continue
        n_with_fp += 1

        expected_source_ids = {exp.replace("\\", "/").split("/")[-2] for exp in expected if "/" in exp}

        print(f"--- {q['category']:<16} attendu={expected}")
        print(f"    Q: {q['question'][:90]}")
        for r in false_positives:
            p = r.get("payload", {})
            source_path = p.get("source_path", "?")
            source_id = p.get("source_id", "?")
            hierarchy_path = p.get("hierarchy_path", "?")
            version_order = p.get("version_order")

            kind = "meme_source_id_autre_fichier" if source_id in expected_source_ids else "source_id_completement_differente"
            fp_kind_counter[kind] += 1

            print(f"      [{kind}] {source_path}  (source_id={source_id}, version_order={version_order})")
            print(f"        hierarchy_path={hierarchy_path}")
        print()

    print("=== Résumé ===")
    print(f"Questions sans faux positif dans le top-{TOP_K} : {n_perfect}/{len(questions)}")
    print(f"Questions avec au moins un faux positif        : {n_with_fp}/{len(questions)}")
    print("Répartition des faux positifs par type :")
    for kind, count in fp_kind_counter.most_common():
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
