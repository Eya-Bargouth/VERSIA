#!/usr/bin/env python3
"""Calibration réelle de AbstentionGate.threshold_low (spec §8) sur données
réelles : fait tourner le vrai pipeline (Qdrant + BGE-M3 + reranker + Ollama)
sur les questions in-corpus (data/eval/questions_v1.jsonl, ne devraient pas
être abstenues) + hors-corpus (data/eval/questions_abstention.jsonl,
abstention attendue), collecte les 4 ingrédients bruts d'
AbstentionGate.evaluate(), puis balaie threshold_low hors-ligne (aucun
nouvel appel LLM) pour mesurer faux-positifs d'abstention vs abstention
correcte à chaque seuil candidat.

"""
import argparse
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
from src.generation.generator import Generator
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.pipeline import QueryPipeline
from src.reliability.abstention import AbstentionGate
from src.reliability.sufficiency import SufficiencyChecker, SufficiencyVerdict
from src.retrieval.hybrid_retriever import HybridRetriever

OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "abstention_calibration_raw.jsonl"
IN_CORPUS_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_v1.jsonl"
OUT_OF_CORPUS_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_abstention.jsonl"


def _load_questions(path: Path, should_abstain: bool, exclude_categories: set[str] = frozenset()) -> list[tuple[dict, bool]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [(row, should_abstain) for row in rows if row.get("category") not in exclude_categories]


def collect() -> list[dict]:
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

    gen_config = LLMConfig(
        provider=settings.llm_provider, model=settings.llm_model, base_url=settings.llm_base_url,
        temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens, timeout=settings.llm_timeout,
        repeat_penalty=settings.llm_repeat_penalty,
    )
    gen_client = LLMFactory.create(gen_config)
    generator = Generator(llm_client=gen_client, llm_config=gen_config, store=store)
    sufficiency_checker = SufficiencyChecker(llm_client=gen_client)

    # IN_CORPUS_PATH (questions_v1.jsonl) contient déjà les questions
    # d'abstention (category="abstention", même contenu que
    # OUT_OF_CORPUS_PATH) — les exclure ici pour ne pas les compter deux
    # fois avec des étiquettes should_abstain contradictoires (bug réel
    # trouvé et corrigé après la première calibration, voir docstring de
    # src/reliability/abstention.py).
    questions = (
        _load_questions(IN_CORPUS_PATH, False, exclude_categories={"abstention"})
        + _load_questions(OUT_OF_CORPUS_PATH, True)
    )
    print(f"=== {len(questions)} questions à traiter "
          f"({sum(1 for _, a in questions if not a)} in-corpus + {sum(1 for _, a in questions if a)} hors-corpus) ===")

    results = []
    for i, (q, should_abstain) in enumerate(questions):
        t0 = time.perf_counter()
        try:
            retrieval = retriever.retrieve(
                query=q["question"], top_k=10, k_dense=settings.retrieval_k_dense, k_sparse=settings.retrieval_k_sparse
            )
            chunks = retrieval["results"]
            diff_explanation = QueryPipeline._diff_explanation(retrieval)
            sufficiency = sufficiency_checker.check(q["question"], chunks, gen_config, diff_explanation=diff_explanation)
            generation = generator.generate(q["question"], chunks, diff_explanation=diff_explanation)
            reranker_scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]

            results.append({
                "question": q["question"],
                "should_abstain": should_abstain,
                "reranker_scores": reranker_scores,
                "generation_confidence": generation.confidence,
                "sufficiency_verdict": sufficiency.verdict,
                "sufficiency_confidence": sufficiency.confidence,
                "planner_confidence": retrieval.get("confidence"),
            })
            elapsed = time.perf_counter() - t0
            print(f"[{i + 1}/{len(questions)}] ({elapsed:.1f}s) should_abstain={should_abstain} "
                  f"suff={sufficiency.verdict} gen_conf={generation.confidence:.2f} -> {q['question'][:60]}")
        except Exception as exc:
            print(f"[{i + 1}/{len(questions)}] ERREUR ({exc}) -> {q['question'][:60]}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(results)}/{len(questions)} questions réussies, écrites dans {OUT_PATH}")
    return results


def sweep(records: list[dict]) -> None:
    in_corpus = [r for r in records if not r["should_abstain"]]
    out_corpus = [r for r in records if r["should_abstain"]]
    print(f"\n{len(in_corpus)} in-corpus, {len(out_corpus)} hors-corpus\n")

    def evaluate_at(threshold_low: float, threshold_high: float) -> tuple[float, float, float]:
        gate = AbstentionGate(threshold_low=threshold_low, threshold_high=threshold_high)
        false_abstention = correct_abstention = 0
        for r in records:
            verdict = SufficiencyVerdict(
                verdict=r["sufficiency_verdict"], confidence=r["sufficiency_confidence"], method="llm_local"
            )
            decision = gate.evaluate(
                reranker_scores=r["reranker_scores"],
                generation_confidence=r["generation_confidence"],
                sufficiency_verdict=verdict,
                planner_confidence=r["planner_confidence"],
            )
            abstained = decision.action == "abstain"
            if r["should_abstain"] and abstained:
                correct_abstention += 1
            elif not r["should_abstain"] and abstained:
                false_abstention += 1
        fa_rate = false_abstention / len(in_corpus) if in_corpus else 0.0
        ca_rate = correct_abstention / len(out_corpus) if out_corpus else 0.0
        accuracy = ((len(in_corpus) - false_abstention) + correct_abstention) / len(records)
        return fa_rate, ca_rate, accuracy

    fa, ca, acc = evaluate_at(0.4, 0.7)
    print(f"Seuils 0.4/0.7 : faux_abstention={fa:.0%} abstention_correcte={ca:.0%} exactitude={acc:.0%}\n")

    for tl in [round(x * 0.05, 2) for x in range(2, 14)]:
        fa, ca, acc = evaluate_at(tl, 0.7)
        print(f"threshold_low={tl:.2f}  faux_abstention={fa:.0%}  abstention_correcte={ca:.0%}  exactitude={acc:.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collect", action="store_true", help="Réanalyse data/eval/abstention_calibration_raw.jsonl sans refaire tourner le pipeline")
    args = parser.parse_args()

    if args.skip_collect:
        with open(OUT_PATH, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    else:
        records = collect()

    sweep(records)


if __name__ == "__main__":
    main()
