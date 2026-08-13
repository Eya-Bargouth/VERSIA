"""
Benchmark qualité retrieval : Recall@K, MRR, nDCG
Comparaison : Dense seul / Sparse seul / Hybrid+RRF / Hybrid+RRF+Reranker
"""

import sys
import os

# Ensure the project root is on sys.path so `from src.xxx import ...` works
# whether the script is executed from the project root, from scripts/, or via
# a task runner — regardless of how the package was installed.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.retrieval.dense_search import DenseSearch
from src.retrieval.sparse_search import SparseSearch
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


# ============================================================================
# 15 Test Queries avec expected_sources (chunks pertinents connus)
# ============================================================================

TEST_QUERIES = [
    {
        "query": "POST /v1/orders parameters",
        "expected_sources": ["stripe_specs", "binance_spot"],  # Peut venir de Stripe ou Binance docs
        "intent": "navigational"
    },
    {
        "query": "How to authenticate API request",
        "expected_sources": ["stripe_specs", "alpaca_incidents"],
        "intent": "factual"
    },
    {
        "query": "Rate limiting error handling",
        "expected_sources": ["stripe_specs", "owasp_cheatsheets"],
        "intent": "factual"
    },
    {
        "query": "Webhook signature verification",
        "expected_sources": ["stripe_specs", "owasp_cheatsheets"],
        "intent": "factual"
    },
    {
        "query": "GET /users endpoint deprecated vs v2",
        "expected_sources": ["stripe_specs", "binance_spot"],
        "intent": "comparative"
    },
    {
        "query": "SQL injection prevention",
        "expected_sources": ["owasp_cheatsheets"],
        "intent": "factual"
    },
    {
        "query": "CORS policy implementation",
        "expected_sources": ["owasp_cheatsheets"],
        "intent": "factual"
    },
    {
        "query": "Binance spot vs futures differences",
        "expected_sources": ["binance_spot"],
        "intent": "comparative"
    },
    {
        "query": "Token refresh mechanism",
        "expected_sources": ["stripe_specs", "alpaca_incidents", "binance_spot"],
        "intent": "factual"
    },
    {
        "query": "Error response format",
        "expected_sources": ["stripe_specs", "binance_spot"],
        "intent": "factual"
    },
    {
        "query": "Alpaca trading halt incidents",
        "expected_sources": ["alpaca_incidents"],
        "intent": "factual"
    },
    {
        "query": "XSS attack mitigation",
        "expected_sources": ["owasp_cheatsheets"],
        "intent": "factual"
    },
    {
        "query": "Pagination strategy large datasets",
        "expected_sources": ["stripe_specs", "binance_spot"],
        "intent": "factual"
    },
    {
        "query": "Backwards compatibility deprecated endpoints",
        "expected_sources": ["stripe_specs"],
        "intent": "comparative"
    },
    {
        "query": "Cryptographic key storage best practices",
        "expected_sources": ["owasp_cheatsheets"],
        "intent": "factual"
    },
]


def is_relevant(payload: Dict[str, Any], expected_sources: List[str]) -> bool:
    """Check if a result chunk matches expected source."""
    source_id = payload.get("source_id", "")
    return any(exp in source_id for exp in expected_sources)


def compute_recall_at_k(results: List[Dict[str, Any]], expected_sources: List[str], k: int) -> float:
    """Recall@K: % of relevant docs in top K."""
    top_k_results = results[:k]
    relevant_count = sum(1 for r in top_k_results if is_relevant(r.get("payload", {}), expected_sources))
    
    # If no relevant docs expected, perfect recall
    if not expected_sources:
        return 1.0
    
    return relevant_count / min(len(expected_sources), k)


def compute_mrr(results: List[Dict[str, Any]], expected_sources: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for i, r in enumerate(results, 1):
        if is_relevant(r.get("payload", {}), expected_sources):
            return 1.0 / i
    return 0.0  # No relevant doc found


def compute_ndcg(results: List[Dict[str, Any]], expected_sources: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain@K.
    
    DCG@K = Σ (rel_i / log2(i+1))
    IDCG@K = Σ (1 / log2(i+1)) for first K relevant docs
    nDCG@K = DCG@K / IDCG@K
    """
    top_k = results[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, r in enumerate(top_k, 1):
        rel = 1.0 if is_relevant(r.get("payload", {}), expected_sources) else 0.0
        dcg += rel / np.log2(i + 1)
    
    # Compute IDCG (ideal: all relevant docs first)
    num_relevant = min(len(expected_sources), k)
    idcg = 0.0
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def benchmark_strategy(
    retriever: HybridRetriever,
    store: QdrantStore,
    embedder: BGEEmbedder,
    test_queries: List[Dict[str, Any]],
    strategy_name: str,
    use_reranker: bool = False,
    k_values: List[int] = [10, 20]
) -> Dict[str, Any]:
    """Run benchmark for one strategy."""

    results_by_k = {k: [] for k in k_values}
    mrr_scores = []

    print(f"\n{'='*70}")
    print(f"  Strategy: {strategy_name} | Reranker: {use_reranker}")
    print(f"{'='*70}")

    for q_idx, q_data in enumerate(test_queries, 1):
        query = q_data["query"]
        expected_sources = q_data["expected_sources"]

        # Real BGE-M3 dense + sparse embeddings for this query (same model
        # used at ingestion time — a random/dummy vector would measure
        # nothing meaningful about actual retrieval quality).
        query_embedding, query_sparse = embedder.embed_query(query)

        # Execute retrieval
        try:
            if "dense_only" in strategy_name.lower():
                results = retriever.dense_search.search_dense(query_embedding, k=50)
            elif "sparse_only" in strategy_name.lower():
                results = retriever.sparse_search.search_sparse(query_sparse, k=50)
            else:
                # Hybrid
                ret = retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    query_sparse=query_sparse,
                    top_k=50,
                    k_dense=50,
                    k_sparse=50
                )
                results = [
                    type('obj', (), {
                        'chunk_id': r['chunk_id'],
                        'score': r['score'],
                        'payload': r['payload']
                    })()
                    for r in ret.get('results', [])
                ]
            
            # Convert to dicts
            result_dicts = [
                {
                    'chunk_id': str(r.chunk_id if hasattr(r, 'chunk_id') else r.get('chunk_id')),
                    'score': r.score if hasattr(r, 'score') else r.get('score'),
                    'payload': r.payload if hasattr(r, 'payload') else r.get('payload', {})
                }
                for r in results
            ]
            
            # Rerank if needed
            if use_reranker:
                result_dicts = retriever.reranker.rerank(query, result_dicts, top_k=50)
            
            # Compute metrics
            mrr = compute_mrr(result_dicts, expected_sources)
            mrr_scores.append(mrr)
            
            for k in k_values:
                recall_k = compute_recall_at_k(result_dicts, expected_sources, k)
                ndcg_k = compute_ndcg(result_dicts, expected_sources, k)
                results_by_k[k].append({
                    'recall': recall_k,
                    'ndcg': ndcg_k
                })
            
            print(f"  Q{q_idx:2d}: MRR={mrr:.3f} | Recall@10={results_by_k[10][-1]['recall']:.3f} | nDCG@10={results_by_k[10][-1]['ndcg']:.3f}")
        
        except Exception as e:
            print(f"  Q{q_idx:2d}: ERROR - {str(e)[:60]}")
            for k in k_values:
                results_by_k[k].append({'recall': 0.0, 'ndcg': 0.0})
            mrr_scores.append(0.0)
    
    # Aggregate metrics
    avg_metrics = {}
    for k in k_values:
        recalls = [m['recall'] for m in results_by_k[k]]
        ndcgs = [m['ndcg'] for m in results_by_k[k]]
        avg_metrics[f'recall@{k}'] = np.mean(recalls)
        avg_metrics[f'ndcg@{k}'] = np.mean(ndcgs)
    
    avg_metrics['mrr'] = np.mean(mrr_scores)
    
    return avg_metrics


def main():
    """Run full benchmark."""
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    collection = os.environ.get('QDRANT_COLLECTION_NAME', 'trade_chunks')
    
    try:
        client = QdrantClient(host=host, port=port)
        store = QdrantStore(client=client, collection_name=collection)
        store.ensure_collection()
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant at {host}:{port}")
        print(f"Details: {e}")
        print("\nTo run this benchmark:")
        print("  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  2. Ingest data: python scripts/ingest_data.py")
        print("  3. Run benchmark: python scripts/benchmark_quality_metrics.py")
        return
    
    # Real BGE-M3 embedder (dense + sparse) — same model/settings as ingestion.
    settings = get_settings()
    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        dtype=settings.embedding_dtype,
        backend=settings.embedding_backend,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    # Create retrievers
    dense = DenseSearch(store)
    sparse = SparseSearch(store)

    retriever_no_rerank = HybridRetriever(store, use_reranker=False)
    retriever_with_rerank = HybridRetriever(store, use_reranker=True)

    # Run benchmarks
    results = {}

    results['dense_only'] = benchmark_strategy(
        retriever_no_rerank, store, embedder, TEST_QUERIES,
        strategy_name="Dense Only",
        use_reranker=False
    )

    results['sparse_only'] = benchmark_strategy(
        retriever_no_rerank, store, embedder, TEST_QUERIES,
        strategy_name="Sparse Only",
        use_reranker=False
    )

    results['hybrid_rrf'] = benchmark_strategy(
        retriever_no_rerank, store, embedder, TEST_QUERIES,
        strategy_name="Hybrid + RRF",
        use_reranker=False
    )

    results['hybrid_rrf_rerank'] = benchmark_strategy(
        retriever_with_rerank, store, embedder, TEST_QUERIES,
        strategy_name="Hybrid + RRF + Reranker",
        use_reranker=True
    )
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Strategy':<30} {'MRR':>10} {'Recall@10':>12} {'nDCG@10':>12}")
    print(f"{'-'*70}")
    
    for strategy, metrics in results.items():
        mrr = metrics.get('mrr', 0.0)
        recall = metrics.get('recall@10', 0.0)
        ndcg = metrics.get('ndcg@10', 0.0)
        print(f"{strategy:<30} {mrr:>10.4f} {recall:>12.4f} {ndcg:>12.4f}")
    
    print(f"\n{'-'*70}")
    print(f"\nFull results:")
    print(json.dumps(results, indent=2))
    
    # Save to file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
