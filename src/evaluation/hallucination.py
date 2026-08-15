"""Hallucination Rate (spec §9).

Auparavant dérivée du signal `support_level` auto-déclaré par le LLM à
chaque citation (Phase 4, style Self-RAG) — mais le prompt système
interdisait explicitement au modèle d'émettre le label `no_support`
("N'ajoute jamais de citation 'no_support' — dans ce cas, ne cite
simplement pas"), ce qui biaisait structurellement la métrique vers 0.0 quel
que soit le taux d'hallucination réel (0.0 rapporté vs Faithfulness du juge
indépendant à 0.448 sur les mêmes réponses — écart mesuré en Phase 5). Le
prompt a été corrigé (le modèle peut maintenant déclarer honnêtement
`no_support`), mais le signal reste par nature de l'auto-évaluation.

Dérivée maintenant du score Faithfulness du juge indépendant (voir
`src/evaluation/ragas_eval.py::faithfulness` — décomposition en claims +
vérification NLI de chacune contre le contexte, pas un jugement du même
modèle qui a produit la réponse) plutôt que d'une auto-déclaration.

Une réponse est considérée « hallucinée » si le système a effectivement
répondu (au moins une citation, donc pas une abstention — voir
QueryPipeline._finalize_answer qui vide `citations` sur abstention) et que
son score Faithfulness indépendant tombe sous `threshold`.
"""

from __future__ import annotations

from typing import Protocol


class _ResultLike(Protocol):
    citations: list
    faithfulness: float


def hallucination_rate(results: list[_ResultLike], threshold: float = 0.5) -> dict:
    answered = [r for r in results if r.citations]
    hallucinated = [r for r in answered if r.faithfulness < threshold]
    return {
        "hallucination_rate": (len(hallucinated) / len(answered)) if answered else 0.0,
        "n_total": len(results),
        "n_answered": len(answered),
        "n_abstained": len(results) - len(answered),
        "n_hallucinated": len(hallucinated),
    }
