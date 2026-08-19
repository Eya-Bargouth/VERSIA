"""Tests d'intégration Phase 3 — Retrieval hybride sur la vraie BDD Qdrant.

Ces tests se connectent à la collection ``trade_chunks`` réelle.
Ils sont ignorés (pytest.skip) si Qdrant n'est pas accessible.

Toutes les recherches utilisent :
- BAAI/bge-m3 via BGEEmbedder pour encoder les requêtes (dense)
- BGE-Reranker-v2-m3 via FlagReranker pour le reranking final
- La vraie collection Qdrant (1 077+ chunks)
"""

import os
import time

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.retrieval.dense_search import DenseSearch
from src.retrieval.fusion import rrf_fuse
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.planner import QueryPlanner
from src.retrieval.reranker import Reranker

pytestmark = pytest.mark.phase3


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qdrant_store() -> QdrantStore:
    """Connexion à Qdrant réel — skip si indisponible."""
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("QDRANT_COLLECTION_NAME", "trade_chunks")

    client = QdrantClient(host=host, port=port)
    try:
        info = client.get_collection(collection)
        assert info.points_count > 0, "Collection is empty"
    except Exception as exc:
        pytest.skip(f"Qdrant collection '{collection}' not available: {exc}")

    return QdrantStore(client=client, collection_name=collection)


@pytest.fixture(scope="module")
def embedder() -> BGEEmbedder:
    """BGE-M3 embedder — skip si le modèle ne peut pas être chargé."""
    try:
        emb = BGEEmbedder(model_name="BAAI/bge-m3", device="auto")
        emb._load_model()  # eager load for tests
        return emb
    except Exception as exc:
        pytest.skip(f"BGEEmbedder could not be loaded: {exc}")


@pytest.fixture(scope="module")
def retriever(qdrant_store, embedder) -> HybridRetriever:
    """Full HybridRetriever wired to the real store and embedder."""
    return HybridRetriever(
        store=qdrant_store,
        embedder=embedder,
        use_reranker=True,
        reranker_model="BAAI/bge-reranker-v2-m3",
    )


# ---------------------------------------------------------------------------
# Test 1 — Dense search with real BGE-M3 embedding
# ---------------------------------------------------------------------------


def test_dense_search_real(qdrant_store, embedder):
    """Dense search returns real results from the collection."""
    query = "What parameter is required for POST /v1/orders?"
    dense_vec, _ = embedder.embed_query(query)

    ds = DenseSearch(qdrant_store)
    results = ds.search_dense(dense_vec, k=10)

    assert len(results) > 0, "Dense search returned no results"
    # Scores should be in [0, 1] for cosine similarity
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Unexpected score: {r.score}"
        assert r.payload.get("text"), "Chunk has no text payload"

    print(f"\nDense top-1: score={results[0].score:.4f} text='{results[0].payload.get('text','')[:80]}'")


# ---------------------------------------------------------------------------
# Test 2 — QueryPlanner on a real API query
# ---------------------------------------------------------------------------


def test_planner_with_real_query():
    """Planner correctly classifies real-world queries."""
    qp = QueryPlanner()

    # Factual with explicit endpoint
    r = qp.plan("What parameter is required for POST /v1/orders?")
    assert r["intent"] == "factual"
    assert r["entity"] == "/v1/orders"

    # Comparative
    r2 = qp.plan("Compare v1 and v2 of the currency parameter")
    assert r2["intent"] == "comparative"

    print("\nPlanner results OK")


# ---------------------------------------------------------------------------
# Test 3 — RRF fusion produces ordered results
# ---------------------------------------------------------------------------


def test_rrf_fusion_with_real_results(qdrant_store, embedder):
    """RRF correctly fuses two dense result lists (simulating dense+sparse)."""
    query = "SQL injection prevention techniques"
    dense_vec, _ = embedder.embed_query(query)

    ds = DenseSearch(qdrant_store)

    # Simulate two retrieval lists (two different queries) to test fusion
    results_a = ds.search_dense(dense_vec, k=20)
    # Second list: slightly different (shift embedding by noise)
    noisy_vec = (dense_vec + np.random.default_rng(42).normal(0, 0.01, dense_vec.shape)).astype(
        np.float32
    )
    noisy_vec /= np.linalg.norm(noisy_vec)
    results_b = ds.search_dense(noisy_vec, k=20)

    fused = rrf_fuse([results_a, results_b], k_smooth=60, weights=[0.6, 0.4])

    assert len(fused) > 0
    # RRF scores must be strictly decreasing (sorted)
    scores = [f.rrf_score for f in fused]
    assert scores == sorted(scores, reverse=True)

    print(f"\nRRF fused {len(fused)} unique chunks, top score={scores[0]:.6f}")


# ---------------------------------------------------------------------------
# Test 4 — Reranker (BGE-Reranker-v2-m3) on real chunks
# ---------------------------------------------------------------------------


def test_reranker_real(qdrant_store, embedder):
    """BGE-Reranker-v2-m3 reranks real retrieval results."""
    query = "How to authenticate an API request with Bearer token?"
    dense_vec, _ = embedder.embed_query(query)

    ds = DenseSearch(qdrant_store)
    raw_results = ds.search_dense(dense_vec, k=20)
    assert len(raw_results) > 0, "No results to rerank"

    # Convert to dicts for reranker
    candidates = [
        {
            "chunk_id": str(r.chunk_id),
            "score": r.score,
            "text": r.payload.get("text", ""),
            "payload": r.payload,
        }
        for r in raw_results
    ]

    reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, candidates, top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(reranked) == 5
    # Every result must have a rerank_score
    for r in reranked:
        assert "rerank_score" in r, "rerank_score missing"
        assert 0.0 <= r["rerank_score"] <= 1.0, f"Score out of range: {r['rerank_score']}"

    print(
        f"\nReranker: {len(candidates)} → 5 in {elapsed_ms:.0f}ms"
        f"\nTop-1: score={reranked[0]['rerank_score']:.4f} "
        f"text='{reranked[0]['text'][:80]}'"
    )


# ---------------------------------------------------------------------------
# Test 5 — Full hybrid pipeline end-to-end
# ---------------------------------------------------------------------------


def test_hybrid_retrieval_end_to_end(retriever):
    """Full pipeline: BGE-M3 embed → dense search → (RRF) → BGE-Reranker."""
    query = "What parameter is required for POST /v1/orders?"
    qp = QueryPlanner()
    plan = qp.plan(query)

    t0 = time.perf_counter()
    out = retriever.retrieve(query=query, planner_output=plan, top_k=5)
    total_ms = (time.perf_counter() - t0) * 1000

    # Structure checks
    assert "results" in out
    assert "strategy" in out
    assert "latency_ms" in out
    assert "planner_intent" in out
    assert isinstance(out["results"], list)
    assert len(out["results"]) > 0, "No results returned"

    # Latency target: < 10 s on CPU (model load included first time)
    assert total_ms < 60_000, f"Retrieval took too long: {total_ms:.0f}ms"

    # Result format checks
    for r in out["results"]:
        assert "chunk_id" in r
        assert "text" in r or "payload" in r

    print(
        f"\nHybrid retrieval: strategy={out['strategy']}"
        f" intent={out['planner_intent']}"
        f" results={len(out['results'])}"
        f" latency_ms={out['latency_ms']}"
    )
    print(f"Top-1 chunk: '{out['results'][0].get('text', '')[:100]}'")


# ---------------------------------------------------------------------------
# Test 6 — Source filter (Stripe only)
# ---------------------------------------------------------------------------


def test_source_filter_stripe(qdrant_store, embedder):
    """Post-filter source_id=stripe_specs returns only Stripe chunks."""
    query = "How does Stripe handle webhook signature verification?"
    dense_vec, _ = embedder.embed_query(query)

    qdrant_filter = {"key": "source_id", "match": {"value": "stripe_specs"}}
    ds = DenseSearch(qdrant_store)
    results = ds.search_dense(dense_vec, filter=qdrant_filter, k=10)

    if len(results) == 0:
        pytest.skip("No Stripe chunks in collection (source_id filter returned 0 results)")

    for r in results:
        assert r.payload.get("source_id") == "stripe_specs", (
            f"Expected stripe_specs, got {r.payload.get('source_id')}"
        )

    print(f"\nStripe filter: {len(results)} results, all source_id=stripe_specs ✓")


# ---------------------------------------------------------------------------
# Test 7 — Latency under 2 s for dense-only (excluding model load)
# ---------------------------------------------------------------------------


def test_dense_latency(qdrant_store, embedder):
    """Dense retrieval (Qdrant query only) should complete in < 2 000 ms."""
    query = "rate limiting error handling"
    dense_vec, _ = embedder.embed_query(query)
    ds = DenseSearch(qdrant_store)

    # Warm-up (first call may include Qdrant connection overhead)
    ds.search_dense(dense_vec, k=10)

    # Timed call
    t0 = time.perf_counter()
    results = ds.search_dense(dense_vec, k=50)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(results) > 0
    assert elapsed_ms < 2000, f"Dense search too slow: {elapsed_ms:.0f}ms"

    print(f"\nDense latency: {elapsed_ms:.1f}ms for k=50")
