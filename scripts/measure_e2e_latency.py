#!/usr/bin/env python3
"""Mesure la latence réelle par étape du pipeline complet : dense search,
sparse search, fusion RRF, reranking (BGE-Reranker-v2-m3), génération LLM
(Ollama), et le total end-to-end. Persiste un artefact JSON horodaté dans
docs/audits/baselines/.

Contrairement à measure_retrieval_latency.py (retrieval seul, vecteur
aléatoire), ce script utilise de vrais embeddings BGE-M3 et exerce toute la
chaîne y compris la génération, pour obtenir des chiffres comparables au
seuil de la spec (< 500ms pour le retrieval seul, hors génération).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig, LLMMessage
from src.llm.providers import ollama_client, vllm_client  # noqa: F401 — self-register
from src.retrieval.fusion import rrf_fuse
from src.retrieval.reranker import Reranker

TEST_QUERIES = [
    "What parameter is required for POST /v1/orders?",
    "How to store passwords securely?",
    "List all orders endpoint",
    "API latency spike incident",
    "Authentication cheat sheet",
]

GEN_PROMPT_TEMPLATE = (
    "Réponds en une phrase à la question suivante en te basant uniquement sur le contexte.\n\n"
    "Contexte:\n{context}\n\nQuestion: {query}\nRéponse:"
)


def _stats(latencies_ms: list) -> dict:
    if not latencies_ms:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "avg_ms": round(sum(s) / n, 1),
        "p50_ms": round(s[n // 2], 1),
        "p95_ms": round(s[min(n - 1, int(n * 0.95))], 1),
    }


def main():
    parser = argparse.ArgumentParser(description="TRADE end-to-end per-stage latency measurement")
    parser.add_argument("--qdrant-url", type=str, default=None)
    parser.add_argument("--skip-generation", action="store_true", help="Skip the LLM generation stage")
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="Override settings.llm_model for this run only (comparison runs).",
    )
    parser.add_argument(
        "--llm-base-url", type=str, default=None,
        help="Override settings.llm_base_url for this run only (e.g. a different local Ollama instance).",
    )
    parser.add_argument(
        "--llm-max-tokens", type=int, default=None,
        help="Override settings.llm_max_tokens for this run only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "audits" / "baselines",
    )
    args = parser.parse_args()

    settings = get_settings()
    qdrant_url = args.qdrant_url or f"http://{settings.qdrant_host}:{settings.qdrant_port}"
    client = QdrantClient(qdrant_url)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    collection_info = client.get_collection(settings.qdrant_collection_name)

    print(f"Collection: {settings.qdrant_collection_name} ({collection_info.points_count} points)")
    effective_model = args.llm_model or settings.llm_model
    effective_base_url = args.llm_base_url or settings.llm_base_url
    if not args.skip_generation:
        print(f"LLM: {effective_model} @ {effective_base_url}")

    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    reranker = Reranker(model_name=settings.reranker_model)

    llm_client = None
    llm_config = None
    if not args.skip_generation:
        llm_config = LLMConfig(
            provider=settings.llm_provider,
            model=args.llm_model or settings.llm_model,
            base_url=args.llm_base_url or settings.llm_base_url,
            temperature=settings.llm_temperature,
            # NOTE: for qwen3:4b, capping this lower (400, then 800) came back
            # with EMPTY content — its internal reasoning for a RAG-style
            # prompt exceeds 800 tokens before it even starts answering. An
            # empty answer is worse than a slow one, so this stays at the
            # configured default unless explicitly overridden (e.g. for a
            # non-thinking model like qwen2.5:7b-instruct, which doesn't need
            # a large budget).
            max_tokens=args.llm_max_tokens or settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )
        llm_client = LLMFactory.create(llm_config)

    # Warm-up: exclude one-time model-load costs from the measured stats.
    print("Warm-up...")
    embedder.embed_query("warm-up query")
    reranker.rerank("warm-up", [{"chunk_id": "0", "text": "warm-up passage", "score": 0.0}], top_k=1)
    if llm_client is not None:
        try:
            llm_client.complete(
                [LLMMessage(role="user", content="Say OK.")], llm_config
            )
        except Exception as exc:
            print(f"  WARNING: LLM warm-up call failed ({exc}); generation stage will likely fail too.")

    stage_latencies = {"embed": [], "dense": [], "sparse": [], "rrf": [], "rerank": [], "generation": [], "total": []}
    run_details = []

    for q in TEST_QUERIES:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        dense_vec, sparse_vec = embedder.embed_query(q)
        t_embed = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense_res = store.search_dense(dense_vec, k=10)
        t_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_res = store.search_sparse(sparse_vec, k=10)
        t_sparse = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse([dense_res, sparse_res], k_smooth=60, weights=[0.6, 0.4])
        t_rrf = (time.perf_counter() - t0) * 1000

        candidates = [
            {"chunk_id": str(f.chunk_id), "score": f.rrf_score, "text": f.payload.get("text", ""), "payload": f.payload}
            for f in fused
        ]

        t0 = time.perf_counter()
        reranked = reranker.rerank(q, candidates, top_k=5)
        t_rerank = (time.perf_counter() - t0) * 1000

        retrieval_only_ms = t_embed + t_dense + t_sparse + t_rrf + t_rerank

        t_gen = None
        gen_content = None
        if llm_client is not None:
            context = "\n\n".join(r.get("text", "")[:500] for r in reranked[:3])
            prompt = GEN_PROMPT_TEMPLATE.format(context=context, query=q)
            t0 = time.perf_counter()
            try:
                resp = llm_client.complete([LLMMessage(role="user", content=prompt)], llm_config)
                t_gen = (time.perf_counter() - t0) * 1000
                gen_content = resp.content[:200]
            except Exception as exc:
                t_gen = (time.perf_counter() - t0) * 1000
                gen_content = f"ERROR: {exc}"

        t_total = (time.perf_counter() - t_start) * 1000

        stage_latencies["embed"].append(t_embed)
        stage_latencies["dense"].append(t_dense)
        stage_latencies["sparse"].append(t_sparse)
        stage_latencies["rrf"].append(t_rrf)
        stage_latencies["rerank"].append(t_rerank)
        if t_gen is not None:
            stage_latencies["generation"].append(t_gen)
        stage_latencies["total"].append(t_total)

        print(
            f"  {q[:45]:45s} embed={t_embed:6.1f} dense={t_dense:6.1f} sparse={t_sparse:6.1f} "
            f"rrf={t_rrf:5.1f} rerank={t_rerank:7.1f} retrieval_only={retrieval_only_ms:7.1f} "
            + (f"gen={t_gen:8.1f}" if t_gen is not None else "gen=skipped")
        )
        run_details.append({
            "query": q,
            "embed_ms": round(t_embed, 1),
            "dense_ms": round(t_dense, 1),
            "sparse_ms": round(t_sparse, 1),
            "rrf_ms": round(t_rrf, 1),
            "rerank_ms": round(t_rerank, 1),
            "retrieval_only_ms": round(retrieval_only_ms, 1),
            "generation_ms": round(t_gen, 1) if t_gen is not None else None,
            "generation_output_preview": gen_content,
            "total_ms": round(t_total, 1),
        })

    retrieval_only_stats = _stats([d["retrieval_only_ms"] for d in run_details])
    print(f"\nRetrieval-only (embed+dense+sparse+rrf+rerank): {retrieval_only_stats}")
    print(f"Spec threshold: < 500ms for retrieval alone -> "
          f"{'PASS' if retrieval_only_stats['p95_ms'] and retrieval_only_stats['p95_ms'] < 500 else 'FAIL'} "
          f"(p95={retrieval_only_stats['p95_ms']}ms)")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": qdrant_url,
        "collection": settings.qdrant_collection_name,
        "collection_points_count": collection_info.points_count,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "reranker_model": settings.reranker_model,
        "llm_provider": settings.llm_provider if llm_client else None,
        "llm_model": effective_model if llm_client else None,
        "llm_base_url": effective_base_url if llm_client else None,
        "generation_skipped": llm_client is None,
        "runs": run_details,
        "stage_stats_ms": {stage: _stats(v) for stage, v in stage_latencies.items()},
        "retrieval_only_stats_ms": retrieval_only_stats,
        "spec_threshold_500ms_retrieval_only": {
            "p95_ms": retrieval_only_stats["p95_ms"],
            "pass": bool(retrieval_only_stats["p95_ms"] and retrieval_only_stats["p95_ms"] < 500),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_tag = (effective_model if not args.skip_generation else "no_generation").replace(":", "-").replace("/", "-")
    out_path = args.output_dir / f"e2e_latency_{model_tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nArtefact persisté : {out_path}")


if __name__ == "__main__":
    main()
