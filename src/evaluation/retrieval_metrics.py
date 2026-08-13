"""Métriques retrieval — Recall@k, Precision@k, MRR, nDCG (spec §9).

Pas de source_id/corpus codé en dur : la pertinence se juge en comparant le
``source_path`` réel de chaque résultat retrouvé à la liste
``expected_sources`` du jeu de test annoté (voir data/eval/questions_v1.jsonl).
"""

from __future__ import annotations

import math
from typing import Any


def _normalize(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def is_relevant(payload: dict[str, Any], expected_sources: list[str]) -> bool:
    """Un résultat est pertinent si son source_path correspond à l'une des
    sources attendues (comparaison par suffixe — tolère un préfixe de
    répertoire différent entre le jeu de test et le payload Qdrant)."""
    source_path = _normalize(payload.get("source_path", ""))
    if not source_path:
        return False
    for expected in expected_sources:
        exp = _normalize(expected)
        if source_path.endswith(exp) or exp.endswith(source_path):
            return True
    return False


def recall_at_k(results: list[dict[str, Any]], expected_sources: list[str], k: int) -> float:
    """Fraction des sources attendues ayant au moins un chunk pertinent
    dans le top-k (pas fraction de chunks — une seule source peut fournir
    plusieurs chunks pertinents sans que ça change le rappel)."""
    if not expected_sources:
        return 1.0
    top_k = results[:k]
    found = {
        exp
        for exp in expected_sources
        if any(is_relevant(r.get("payload", {}), [exp]) for r in top_k)
    }
    return len(found) / len(set(_normalize(e) for e in expected_sources))


def precision_at_k(results: list[dict[str, Any]], expected_sources: list[str], k: int) -> float:
    """Fraction des k résultats retournés qui sont pertinents."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for r in top_k if is_relevant(r.get("payload", {}), expected_sources))
    return relevant_count / len(top_k)


def mrr(results: list[dict[str, Any]], expected_sources: list[str]) -> float:
    """Mean Reciprocal Rank : 1/rang du premier résultat pertinent."""
    for i, r in enumerate(results, 1):
        if is_relevant(r.get("payload", {}), expected_sources):
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[dict[str, Any]], expected_sources: list[str], k: int) -> float:
    """nDCG@k, pertinence binaire (1 si le chunk vient d'une source attendue).

    IDCG = DCG du même ensemble de résultats réordonné pour placer tous les
    éléments pertinents en tête (définition standard) — PAS dérivé du
    nombre de sources attendues : une seule source peut légitimement
    fournir plusieurs chunks pertinents dans le top-k (granularité chunk vs
    granularité fichier), et capper l'IDCG au nombre de sources sous-compte
    l'idéal quand plusieurs chunks pertinents de la même source apparaissent
    — ça faisait dépasser 1.0 (bug trouvé en vérifiant les résultats
    obtenus, pas en le supposant correct)."""
    top_k = results[:k]
    relevances = [1.0 if is_relevant(r.get("payload", {}), expected_sources) else 0.0 for r in top_k]
    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances, 1))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(results: list[dict[str, Any]], expected_sources: list[str], k_values: list[int]) -> dict[str, float]:
    """Toutes les métriques pour une requête, aux k demandés."""
    metrics = {"mrr": mrr(results, expected_sources)}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(results, expected_sources, k)
        metrics[f"precision@{k}"] = precision_at_k(results, expected_sources, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(results, expected_sources, k)
    return metrics


def aggregate(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Moyenne simple par métrique sur toutes les requêtes."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics) for k in keys}
