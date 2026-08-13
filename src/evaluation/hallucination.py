"""Hallucination Rate (spec §9) — dérivée du signal `support_level` déjà
auto-déclaré par le LLM à chaque citation (Phase 4, style Self-RAG), pas
d'appel juge supplémentaire nécessaire pour cette métrique précise.

Une réponse est considérée « halluciné » si le système a effectivement
répondu (au moins une citation, donc pas une abstention — voir
QueryPipeline._finalize_answer qui vide `citations` sur abstention) mais
qu'au moins une de ses citations est auto-évaluée `no_support` : le LLM a
cité un passage à l'appui d'une affirmation tout en indiquant lui-même que
ce passage ne la soutient pas.
"""

from __future__ import annotations

from typing import Protocol


class _CitationLike(Protocol):
    support_level: str


class _ResultLike(Protocol):
    citations: list[_CitationLike]


def hallucination_rate(results: list[_ResultLike]) -> dict:
    answered = [r for r in results if r.citations]
    hallucinated = [r for r in answered if any(c.support_level == "no_support" for c in r.citations)]
    return {
        "hallucination_rate": (len(hallucinated) / len(answered)) if answered else 0.0,
        "n_total": len(results),
        "n_answered": len(answered),
        "n_abstained": len(results) - len(answered),
        "n_hallucinated": len(hallucinated),
    }
