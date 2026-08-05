"""Measure retrieval latency for hybrid retriever (dense+sparse+fusion)"""
import os
import sys

# Project root on sys.path so `from src.xxx import ...` works when run as
# `python scripts/measure_retrieval_latency.py` from the project root (see
# CLAUDE.md: pythonpath=["src"] only applies to pytest, not standalone scripts).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from qdrant_client import QdrantClient
from src.embeddings.vector_store import QdrantStore
from src.retrieval.hybrid_retriever import HybridRetriever
import numpy as np


def main():
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    collection = os.environ.get('QDRANT_COLLECTION_NAME', 'trade_chunks')
    client = QdrantClient(host=host, port=port)
    store = QdrantStore(client=client, collection_name=collection)
    retriever = HybridRetriever(store=store, use_reranker=False)

    q = 'What parameter is required for POST /v1/orders?'
    # Random vector for measurement
    vec = np.random.randn(1024).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    runs = int(os.environ.get('RETRIEVE_RUNS', '20'))
    latencies = []
    for i in range(runs):
        t0 = time.time()
        out = retriever.retrieve(query=q, query_embedding=vec, top_k=10)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    print(json.dumps({
        'runs': runs,
        'p50_ms': latencies[len(latencies)//2],
        'p95_ms': latencies[int(len(latencies)*0.95)],
        'avg_ms': sum(latencies)/len(latencies)
    }, indent=2))

if __name__ == '__main__':
    main()
