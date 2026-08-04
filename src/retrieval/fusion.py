"""RRF Fusion (Reciprocal Rank Fusion) — Client-side fusion of dense and sparse results."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from uuid import UUID

from src.embeddings.vector_store import RetrievalResult


@dataclass
class FusedResult:
    """Fused retrieval result from RRF."""
    chunk_id: Union[str, UUID]
    rrf_score: float
    source_lists: List[str]
    payload: Dict[str, Any]


def rrf_fuse(
    lists: List[List[Any]],
    k_smooth: int = 60,
    weights: Optional[List[float]] = None
) -> List[FusedResult]:
    """Reciprocal Rank Fusion for combining multiple ranked lists.

    RRF score formula: Σ(1 / (k_smooth + rank_i))

    Args:
        lists: List of ranked result lists. Each result can be:
               - RetrievalResult (Pydantic BaseModel)
               - dict with "chunk_id" and "payload" keys
               - Any object with chunk_id and payload attributes
        k_smooth: Smoothing parameter (default 60)
        weights: Optional weights per list (must sum to 1). Default: equal weights.

    Returns:
        Sorted list of FusedResult, highest RRF score first
    """
    if not lists or not any(lists):
        return []
    
    num_lists = len(lists)
    if weights is None:
        weights = [1.0 / num_lists] * num_lists
    elif abs(sum(weights) - 1.0) > 1e-6:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Build RRF scores
    rrf_scores = {}
    source_lists = {}
    payloads = {}

    for list_idx, ranked_list in enumerate(lists):
        weight = weights[list_idx]
        for rank, result in enumerate(ranked_list):
            chunk_id = None
            payload = {}

            # Handle various types: RetrievalResult, dict, SimpleNamespace, or any object with attributes
            if isinstance(result, dict):
                chunk_id = str(result.get("chunk_id") or result.get("id"))
                payload = result.get("payload", {})
            elif isinstance(result, RetrievalResult):
                chunk_id = str(result.chunk_id)
                payload = result.payload
            else:
                # Try attribute access (works for SimpleNamespace, dataclass, custom objects)
                try:
                    chunk_id = str(getattr(result, "chunk_id", None))
                    payload = getattr(result, "payload", {})
                except Exception:
                    continue

            if chunk_id is None or chunk_id == "None":
                continue
            
            score = weight / (k_smooth + rank + 1)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                source_lists[chunk_id] = []
                payloads[chunk_id] = payload
            
            rrf_scores[chunk_id] += score
            source_lists[chunk_id].append(f"list_{list_idx}")

    # Sort by RRF score, descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        FusedResult(
            chunk_id=chunk_id,
            rrf_score=score,
            source_lists=source_lists[chunk_id],
            payload=payloads.get(chunk_id, {})
        )
        for chunk_id, score in sorted_items
    ]
