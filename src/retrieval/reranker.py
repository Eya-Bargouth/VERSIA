"""Reranker for retrieval results — BGE-Reranker-v2-m3 (FlagEmbedding) avec fallback heuristique."""

import time
from typing import List, Dict, Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class Reranker:
    """Rerank retrieval candidates using BGE-Reranker-v2-m3 (FlagEmbedding).

    Lazy-loads the model on first use.  Falls back to a deterministic
    keyword-overlap heuristic if the model cannot be loaded (e.g., no GPU
    memory, model not downloaded yet).

    Args:
        model_name: HuggingFace model id (default: BAAI/bge-reranker-v2-m3)
        use_fp16:   Use half-precision on GPU (ignored on CPU).
        device:     "cpu", "cuda", or "auto" (picks cuda if available).
        batch_size: Number of (query, passage) pairs per inference batch.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model = None          # lazy-loaded
        self._model_failed = False  # permanent flag: don't retry after failure

        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError as exc:
            # torch is an optional dependency for the heuristic-only path:
            # degrade to the deterministic fallback instead of crashing.
            logger.warning("reranker_torch_unavailable", error=str(exc))
            self._model_failed = True
            cuda_available = False

        if device == "auto":
            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = device

        # FP16 only makes sense on GPU
        if self.device == "cpu":
            self.use_fp16 = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Force le chargement du modèle (+ une inférence factice) tout de
        suite plutôt que d'attendre le premier appel utilisateur — la
        latence de chargement (souvent plusieurs secondes) doit être payée
        au démarrage du service, pas mesurée dans la latence de la première
        requête réelle. À appeler explicitement au démarrage (script ou
        API), jamais automatiquement dans __init__ (romprait la rapidité
        des tests unitaires qui construisent un Reranker sans vouloir
        charger le modèle)."""
        model = self._get_model()
        if model is None:
            return
        try:
            model.compute_score([["warm-up", "warm-up"]], normalize=True)
        except Exception as exc:
            logger.warning("reranker_warmup_inference_failed", error=str(exc))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates by relevance to *query*.

        Tries the real BGE-Reranker-v2-m3 model first; falls back to a
        deterministic heuristic if the model is unavailable.

        Args:
            query:      The user query string.
            candidates: List of dicts with at least 'chunk_id' and 'text'
                        (or 'content') keys.  'payload' is preserved.
            top_k:      Maximum number of results to return.

        Returns:
            Re-ranked list (best first) with an extra 'rerank_score' key.
        """
        if not candidates:
            return []

        model = self._get_model()

        if model is not None:
            return self._rerank_with_model(model, query, candidates, top_k)
        else:
            logger.warning(
                "reranker_using_heuristic",
                model=self.model_name,
                reason="model not available",
            )
            return self._rerank_heuristic(query, candidates, top_k)

    # ------------------------------------------------------------------
    # Model loading (lazy, once)
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the loaded FlagReranker, or None if unavailable."""
        if self._model is not None:
            return self._model
        if self._model_failed:
            return None

        t0 = time.perf_counter()
        try:
            from FlagEmbedding import FlagReranker

            logger.info(
                "loading_reranker_model",
                model=self.model_name,
                device=self.device,
                fp16=self.use_fp16,
            )
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("reranker_model_loaded", model=self.model_name, load_ms=round(elapsed))
            return self._model

        except Exception as exc:
            self._model_failed = True
            logger.warning(
                "reranker_model_load_failed",
                model=self.model_name,
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Real model reranking
    # ------------------------------------------------------------------

    def _rerank_with_model(
        self,
        model,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Score (query, passage) pairs with the real CrossEncoder model."""
        texts = [c.get("text") or c.get("content", "") for c in candidates]

        # Build (query, passage) pairs
        pairs = [[query, t] for t in texts]

        # Batch inference
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = model.compute_score(batch, normalize=True)
            # compute_score can return a single float when batch size == 1
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        # Pair candidates with their scores and sort
        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for cand, score in scored[:top_k]:
            cand = dict(cand)  # copy — don't mutate caller's dicts
            cand["rerank_score"] = float(score)
            reranked.append(cand)

        return reranked

    # ------------------------------------------------------------------
    # Heuristic fallback (deterministic, no model needed)
    # ------------------------------------------------------------------

    def _rerank_heuristic(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fallback: keyword overlap + original retrieval score + hierarchy bonus."""
        query_tokens = set(query.lower().split())
        query_keywords = [w for w in query_tokens if len(w) > 3]

        def score_candidate(cand: Dict[str, Any]) -> float:
            base_score = float(cand.get("score", 0.0))
            text = (cand.get("text") or cand.get("content", "")).lower()
            text_tokens = set(text.split())
            keyword_overlap = sum(1 for kw in query_keywords if kw in text_tokens)
            hierarchy_bonus = 0.05 if cand.get("payload", {}).get("hierarchy_path") else 0.0
            return base_score + (keyword_overlap * 0.01) + hierarchy_bonus

        scored = [(cand, score_candidate(cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for cand, score in scored[:top_k]:
            cand = dict(cand)
            cand["rerank_score"] = score
            reranked.append(cand)

        return reranked
