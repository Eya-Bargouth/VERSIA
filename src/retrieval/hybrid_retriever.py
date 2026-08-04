"""Hybrid retriever — orchestrates dense (+sparse) + RRF + BGE-reranker."""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from src.embeddings.vector_store import QdrantStore, RetrievalResult
from src.ingestion.version_diff import DiffReport, VersionDiffEngine
from src.retrieval.dense_search import DenseSearch
from src.retrieval.fusion import FusedResult, rrf_fuse
from src.retrieval.planner import QueryPlanner
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_search import SparseSearch

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """Hybrid dense+sparse+RRF retriever with filtering and reranking.

    Takes a ``QdrantStore`` as its single external dependency.
    ``DenseSearch`` and ``SparseSearch`` are created internally so callers
    don't have to wire them manually.

    Args:
        store:        Configured QdrantStore (wraps qdrant_client).
        embedder:     Optional BGEEmbedder instance.  When provided, queries
                      are embedded on-the-fly and both dense *and* sparse
                      vectors are computed.  When None, callers must pass
                      pre-computed vectors to ``retrieve()``.
        use_reranker: Whether to rerank the fused candidates with BGE-Reranker.
        reranker_model: HuggingFace id for the reranker (default: v2-m3).
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder=None,
        llm_client=None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client
        self.use_reranker = use_reranker

        self.planner = QueryPlanner(llm_client=llm_client)
        self.dense_search = DenseSearch(store)
        self.sparse_search = SparseSearch(store)
        self.reranker = Reranker(model_name=reranker_model)
        self.version_diff_engine = VersionDiffEngine()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        query_sparse: Optional[Dict[int, float]] = None,
        planner_output: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        k_dense: int = 50,
        k_sparse: int = 50,
    ) -> Dict[str, Any]:
        """Hybrid retrieval: plan → dense/sparse → RRF → rerank.

        Args:
            query:           User query string.
            query_embedding: Optional pre-computed dense embedding (1-D float32
                             array, dim 1024).  Computed via *embedder* if None.
            query_sparse:    Optional pre-computed sparse weights
                             ``{token_id: weight}``.  Computed via *embedder*
                             when available and *query_sparse* is None.
            planner_output:  Optional output of ``QueryPlanner.plan()``.
            top_k:           Final number of results returned.
            k_dense:         Dense candidates retrieved from Qdrant.
            k_sparse:        Sparse candidates retrieved from Qdrant.

        Returns:
            dict with keys:
                results          – list of top_k result dicts
                total_candidates – number of candidates before reranking
                strategy         – "dense", "sparse", or "hybrid"
                planner_intent   – intent label from planner
                planner_entity   – extracted entity (path / endpoint) or None
                latency_ms       – wall-clock time for the full retrieve() call
                confidence       – "low" when intent is "ambiguous" (else absent)
                diff_available   – present only when intent is "comparative"
                diff_report      – precomputed DiffReport dict, when available
                                    and intent is "comparative"

        Intent-driven strategy (spec §7 "Détails Query Planner"):
            factual + entity   → vector search, then post-filter on hierarchy_path
            factual, no entity → vector search only
            comparative        → vector search for context + precomputed diff JSON
                                  (via VersionDiffEngine.load_diff), if available
            navigational       → vector search filtered by source_id (already
                                  applied via the Qdrant filter built above)
            ambiguous          → vector search only + confidence="low" flag
        """
        t_start = time.perf_counter()

        # ── 1. Planning ────────────────────────────────────────────────
        if planner_output is None:
            planner_output = self.planner.plan(query)

        intent = planner_output.get("intent", "ambiguous")
        entity = planner_output.get("entity")
        filters = planner_output.get("filters", {})

        # ── 2. Build Qdrant filter ─────────────────────────────────────
        # Only filter on fields that are explicitly set (non-None).
        # The collection was ingested with status=None on many chunks, so we
        # must NOT apply a status filter unless the user specifically asked for
        # a particular status (e.g. "deprecated").
        qdrant_filter = self._build_qdrant_filter(filters, query)

        # ── 3. Compute embeddings if not provided ─────────────────────
        if query_embedding is None and self.embedder is not None:
            dense_vec, sparse_vec = self.embedder.embed_query(query)
            query_embedding = dense_vec
            if query_sparse is None:
                query_sparse = sparse_vec

        # ── 4. Execute search ─────────────────────────────────────────
        results: List = []
        strategy = "hybrid"

        if query_embedding is not None and query_sparse is not None:
            # Both vectors available → full hybrid
            results = self._hybrid_search(
                query_embedding, query_sparse, qdrant_filter, k_dense, k_sparse, top_k
            )
        elif query_embedding is not None:
            # Dense-only (sparse not available or not computed)
            results = self.dense_search.search_dense(
                query_embedding, filter=qdrant_filter, k=k_dense
            )
            strategy = "dense"
        elif query_sparse is not None:
            # Sparse-only
            results = self.sparse_search.search_sparse(
                query_sparse, filter=qdrant_filter, k=k_sparse
            )
            strategy = "sparse"
        else:
            # No embeddings at all — log and return empty
            logger.warning(
                "hybrid_retriever_no_embeddings",
                hint="Pass query_embedding or attach an embedder= to HybridRetriever",
            )
            return {
                "results": [],
                "total_candidates": 0,
                "strategy": "none",
                "planner_intent": intent,
                "planner_entity": entity,
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
            }

        # ── 5. Normalise to dicts ─────────────────────────────────────
        candidates = self._to_dicts(results)

        # ── 6. Intent-driven strategy refinements ──────────────────────
        extra: Dict[str, Any] = {}
        if intent == "factual" and entity:
            # factual + entity → narrow to chunks under that hierarchy_path
            candidates = self._post_filter_hierarchy(candidates, entity)
        elif intent == "comparative":
            # comparative → vector context stays as-is, plus a precomputed
            # deterministic diff if one is available for the versions mentioned
            diff_report = self._load_comparative_diff(query, filters)
            extra["diff_available"] = diff_report is not None
            if diff_report is not None:
                extra["diff_report"] = diff_report.model_dump()
        elif intent == "navigational":
            # navigational → source_id filter was already applied to the Qdrant
            # query above (via qdrant_filter); no further narrowing here.
            pass
        elif intent == "ambiguous":
            # ambiguous → vector-only, flagged so callers can lower trust /
            # ask a clarifying question downstream.
            extra["confidence"] = "low"

        total_before_rerank = len(candidates)

        # ── 7. Rerank ─────────────────────────────────────────────────
        if self.use_reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        logger.info(
            "hybrid_retrieval_done",
            strategy=strategy,
            intent=intent,
            candidates_before_rerank=total_before_rerank,
            returned=len(candidates),
            latency_ms=latency_ms,
        )

        return {
            "results": candidates,
            "total_candidates": total_before_rerank,
            "strategy": strategy,
            "planner_intent": intent,
            "planner_entity": entity,
            "latency_ms": latency_ms,
            **extra,
        }

    # ------------------------------------------------------------------
    # Parallel dense + sparse search → RRF
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_sparse: Dict[int, float],
        qdrant_filter: Optional[Dict[str, Any]],
        k_dense: int,
        k_sparse: int,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Run dense and sparse searches in parallel, then fuse with RRF."""
        dense_results: List = []
        sparse_results: List = []

        def dense_task():
            nonlocal dense_results
            dense_results = self.dense_search.search_dense(
                query_embedding, filter=qdrant_filter, k=k_dense
            )

        def sparse_task():
            nonlocal sparse_results
            try:
                sparse_results = self.sparse_search.search_sparse(
                    query_sparse, filter=qdrant_filter, k=k_sparse
                )
            except Exception as exc:
                # Collection may not have a sparse index — degrade gracefully
                logger.warning("sparse_search_failed", error=str(exc))
                sparse_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(dense_task)
            executor.submit(sparse_task)

        # If sparse came back empty, return dense results directly (no RRF needed)
        if not sparse_results:
            return dense_results[:top_k]

        fused: List[FusedResult] = rrf_fuse(
            [dense_results, sparse_results],
            k_smooth=60,
            weights=[0.6, 0.4],  # slightly favour dense for this corpus
        )

        return [
            RetrievalResult(
                chunk_id=f.chunk_id,
                score=f.rrf_score,
                source="hybrid",
                payload=f.payload,
            )
            for f in fused[:top_k]
        ]

    # ------------------------------------------------------------------
    # Qdrant filter builder
    # ------------------------------------------------------------------

    def _build_qdrant_filter(
        self, filters: Dict[str, Any], query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Build a Qdrant filter dict from planner filters.

        Rules:
        - ``status`` is only added when the query *explicitly* mentions a
          status keyword (active/deprecated/superseded) — never added as a
          silent default, because the current collection has status=None on
          most chunks.
        - ``source_id`` and ``version_tag`` are added when present.
        - ``valid_from`` / ``valid_until`` are ignored if None.
        """
        if not filters:
            return None

        must_conditions = []
        query_lower = query.lower()

        # Status — only when explicitly requested
        explicit_status = (
            "active" in query_lower
            or "deprecated" in query_lower
            or "obsolete" in query_lower
            or "superseded" in query_lower
        )
        status_val = filters.get("status")
        if explicit_status and status_val:
            must_conditions.append(
                {"key": "status", "match": {"value": status_val}}
            )

        if filters.get("source_id"):
            must_conditions.append(
                {"key": "source_id", "match": {"value": filters["source_id"]}}
            )

        if filters.get("version_tag"):
            must_conditions.append(
                {"key": "version_tag", "match": {"value": filters["version_tag"]}}
            )

        if filters.get("valid_from"):
            must_conditions.append(
                {"key": "valid_from", "range": {"gte": filters["valid_from"]}}
            )

        if filters.get("valid_until"):
            must_conditions.append(
                {"key": "valid_until", "range": {"lte": filters["valid_until"]}}
            )

        if not must_conditions:
            return None

        return (
            {"must": must_conditions}
            if len(must_conditions) > 1
            else must_conditions[0]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dicts(results: List) -> List[Dict[str, Any]]:
        """Normalise heterogeneous result types to plain dicts."""
        out = []
        for r in results:
            if isinstance(r, dict):
                out.append(r)
            elif isinstance(r, RetrievalResult):
                out.append(
                    {
                        "chunk_id": str(r.chunk_id),
                        "score": r.score,
                        "text": r.payload.get("text", ""),
                        "payload": r.payload,
                    }
                )
            else:
                # SimpleNamespace, dataclass, FusedResult …
                out.append(
                    {
                        "chunk_id": str(getattr(r, "chunk_id", "")),
                        "score": float(getattr(r, "score", 0.0)),
                        "text": getattr(r, "payload", {}).get("text", "")
                        if hasattr(r, "payload")
                        else "",
                        "payload": getattr(r, "payload", {}),
                    }
                )
        return out

    def _load_comparative_diff(
        self, query: str, filters: Dict[str, Any]
    ) -> Optional[DiffReport]:
        """Best-effort lookup of a precomputed diff for a comparative query.

        Looks for two of the planner's known version tags mentioned in the
        query text, plus a source_id (from the planner's filters), and
        delegates to ``VersionDiffEngine.load_diff``. Returns None — never
        raises — when the tags/source can't be resolved or no diff has been
        precomputed yet for that pair; callers fall back to vector-only
        context, per spec §7.
        """
        source_id = filters.get("source_id")
        if not source_id:
            return None

        query_lower = query.lower()
        tags_in_query = [
            tag for tag in self.planner.version_tags if tag and tag.lower() in query_lower
        ]
        if len(tags_in_query) < 2:
            return None

        version_from, version_to = tags_in_query[0], tags_in_query[1]
        try:
            return self.version_diff_engine.load_diff(source_id, version_from, version_to)
        except Exception as exc:
            logger.warning(
                "comparative_diff_load_failed",
                source_id=source_id,
                version_from=version_from,
                version_to=version_to,
                error=str(exc),
            )
            return None

    def _post_filter_hierarchy(
        self, candidates: List[Dict[str, Any]], entity: str
    ) -> List[Dict[str, Any]]:
        """Keep only candidates whose hierarchy_path contains *entity*."""
        filtered = [
            c
            for c in candidates
            if entity.lower() in c.get("payload", {}).get("hierarchy_path", "").lower()
        ]
        # If nothing matches, fall back to all candidates (don't lose results)
        return filtered if filtered else candidates
