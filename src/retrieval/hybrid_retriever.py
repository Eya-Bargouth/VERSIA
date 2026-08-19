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
        llm_config=None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.use_reranker = use_reranker

        self.planner = QueryPlanner(llm_client=llm_client, llm_config=llm_config)
        self.dense_search = DenseSearch(store)
        self.sparse_search = SparseSearch(store)
        self.reranker = Reranker(model_name=reranker_model)
        self.version_diff_engine = VersionDiffEngine()

    def warm_up(self) -> None:
        """Charge le reranker immédiatement plutôt qu'au premier retrieve()
        — à appeler explicitement au démarrage (script ou API), pas
        automatiquement ici (romprait la rapidité des tests unitaires)."""
        if self.use_reranker:
            self.reranker.warm_up()

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
        k_dense: int = 15,
        k_sparse: int = 15,
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

        Intent-driven strategy (déviation assumée de la spec §7 "Détails Query
        Planner" — l'intent "navigational" et le filtre source_id automatique
        associé ont été retirés : 0/50 questions réelles du jeu de test ne
        déclenchaient jamais correctement "navigational", et le filtre
        source_id qui lui était réservé s'appliquait en réalité à tous les
        intents sans distinction, cassant le recall des questions
        multi-sources dès qu'une seule des sources attendues était nommée
        dans le texte — voir docs/evaluation_report.md pour la mesure) :
            factual + entity   → vector search, then post-filter on hierarchy_path
            factual, no entity → vector search only
            comparative        → vector search for context + precomputed diff JSON
                                  (via VersionDiffEngine.load_diff), if available
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
            diff_report = self._load_comparative_diff(query, candidates)
            extra["diff_available"] = diff_report is not None
            if diff_report is not None:
                extra["diff_report"] = diff_report.model_dump()
        elif intent == "ambiguous":
            # ambiguous → vector-only, flagged so callers can lower trust /
            # ask a clarifying question downstream.
            extra["confidence"] = "low"

        # "comparative" peut vouloir dire deux choses très différentes : une
        # vraie comparaison de versions (les questions version_conflict,
        # "entre la version X et la version Y" — un vrai tag est alors
        # détecté par le planner) OU une comparaison conceptuelle entre
        # sources sans rapport avec le versioning (ex. "différence entre TLS
        # Client Auth et l'API Stripe", classée comparative par le vocabulaire
        # seul). Dans le second cas, le contenu jugé "identique" entre deux
        # versions Stripe peut coïncidentiellement correspondre à la version
        # que la question attend (ex. "legacy") sans rapport avec la question
        # posée — dédupliquer y ferait perdre cette source sans aucune
        # justification (régression mesurée : Recall@10 sur les questions
        # multi_source/ambiguous, 0.6→0.2, avant ce garde-fou). Ne toucher aux
        # candidats que quand un vrai signal de version a été détecté.
        has_version_signal = bool(filters.get("version_tags") or filters.get("version_tag"))
        if intent != "comparative" or has_version_signal:
            candidates = self._dedupe_version_siblings(candidates, keep_distinct_versions=(intent == "comparative"))

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

        # If sparse came back empty, return dense results directly (no RRF needed).
        # Not truncated to top_k here: the caller reranks this pool before its
        # own final top_k truncation — slicing to top_k this early used to
        # make `len(candidates) > top_k` (the rerank gate in retrieve()) never
        # true, silently skipping reranking on every hybrid-strategy query.
        if not sparse_results:
            return dense_results

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
            for f in fused
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
        - ``version_tag``/``version_tags`` are added when present.
        - ``valid_from`` / ``valid_until`` are ignored if None.

        Pas de filtre ``source_id`` ici : retiré (voir retrieve()) — verrouiller
        la recherche à une seule source dès qu'elle est nommée dans le texte
        cassait le recall des questions multi-sources.
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

        if filters.get("version_tags"):
            must_conditions.append(
                {"key": "version_tag", "match": {"any": filters["version_tags"]}}
            )
        elif filters.get("version_tag"):
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
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Optional[DiffReport]:
        """Best-effort lookup of a precomputed diff for a comparative query.

        Looks for two of the planner's known version tags mentioned in the
        query text, plus a source_id inferred from the payload of the
        candidates actually retrieved (pas du texte de la question — le
        filtre source_id automatique basé sur des mots-clés a été retiré, il
        cassait le recall des questions multi-sources dès qu'une seule des
        sources attendues était nommée dans le texte), et délègue à
        ``VersionDiffEngine.load_diff``. Retourne None — jamais d'exception —
        si les tags/la source ne peuvent pas être résolus ou qu'aucun diff
        n'a été précalculé pour cette paire ; les appelants retombent sur le
        contexte vectoriel seul, per spec §7.
        """
        source_id = next(
            (c.get("payload", {}).get("source_id") for c in candidates if c.get("payload", {}).get("source_id")),
            None,
        )
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

    @staticmethod
    def _normalize_text_for_dedup(text: str) -> str:
        """Tolère les différences d'espacement/retours à la ligne sans
        tolérer un contenu réellement différent — même logique que
        src/generation/citation.py::_normalize_whitespace."""
        return " ".join((text or "").split())

    @classmethod
    def _dedupe_version_siblings(
        cls, candidates: List[Dict[str, Any]], keep_distinct_versions: bool = False
    ) -> List[Dict[str, Any]]:
        """Regroupe les candidats par (source_id, hierarchy_path) au sein
        d'une même source versionnée, puis élimine les doublons de contenu.

        Cas réel constaté (mesure retrieval après le fix #8) : une source
        versionnée dont le contenu ne change pas d'une version à l'autre pour
        une section donnée (ex. un même endpoint identique dans 4 fichiers de
        version Stripe) produit des embeddings quasi identiques — ces
        doublons se marchent dessus dans le top-k au lieu de laisser la place
        à du contenu réellement distinct, dégradant Precision/MRR/nDCG sans
        rien apporter (c'est la même information, en double).

        ``keep_distinct_versions=False`` (par défaut, questions non
        comparatives) : ne garde que la version la plus récente de chaque
        groupe, quel que soit son contenu — une question factuelle veut la
        vérité actuelle, pas un mélange de versions périmées.

        ``keep_distinct_versions=True`` (questions comparatives) : garde la
        version la plus récente, puis conserve aussi toute autre version du
        groupe dont le texte diffère réellement (comparaison possible) —
        élimine seulement les copies au contenu identique, jamais une version
        dont le contenu a changé. Comparer 4 copies d'un texte inchangé
        n'apporte rien, que les 2 versions demandées soient connues ou non
        (voir _build_qdrant_filter/version_tags pour le cas où elles le sont
        — filtré en amont dans Qdrant, avant même d'arriver ici).

        Générique : ne dépend que de métadonnées déjà posées à l'ingestion
        (source_id, hierarchy_path, version_order — voir
        src/dom/builders/base.py::_resolve_version) et du texte du chunk,
        jamais d'un nom de source. Un chunk sans version_order (source non
        versionnée) n'est jamais regroupé avec un autre — chaque tel chunk
        reste sa propre clé unique, garantissant qu'aucun résultat non
        versionné n'est éliminé par erreur.
        """
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        key_order: List[Any] = []
        for candidate in candidates:
            payload = candidate.get("payload", {})
            version_order = payload.get("version_order")
            key = (
                (id(candidate),)
                if version_order is None
                else (payload.get("source_id"), payload.get("hierarchy_path"))
            )
            if key not in groups:
                groups[key] = []
                key_order.append(key)
            groups[key].append(candidate)

        result: List[Dict[str, Any]] = []
        for key in key_order:
            group = groups[key]
            if len(group) == 1:
                result.append(group[0])
                continue

            group_sorted = sorted(
                group,
                key=lambda c: c.get("payload", {}).get("version_order") or -1,
                reverse=True,
            )
            kept = [group_sorted[0]]  # toujours la version la plus récente
            if keep_distinct_versions:
                kept_texts = [cls._normalize_text_for_dedup(kept[0].get("text", ""))]
                for candidate in group_sorted[1:]:
                    text = cls._normalize_text_for_dedup(candidate.get("text", ""))
                    if text and text in kept_texts:
                        continue  # contenu identique à une version déjà gardée
                    kept.append(candidate)
                    kept_texts.append(text)
            result.extend(kept)

        return result

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
