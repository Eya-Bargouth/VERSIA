"""AbstentionGate — 3 seuils CRAG-like (spec §8 "Détails abstention CRAG-like").

Seuils PROVISOIRES (0.4 / 0.7) : la spec demande des seuils "calibrés sur jeu
de test v1", mais ce jeu de test (50 questions annotées) est un livrable de
la Phase 5, pas encore construit — impossible de calibrer avant d'avoir les
données. Décision actée avec l'utilisateur : démarrer avec des valeurs
raisonnables mais arbitraires, marquées comme telles, à recalibrer en
Phase 5 plutôt que de bloquer la Phase 4 dessus.

Note sur `reranker_scores` : `Reranker.rerank()` utilise
`model.compute_score(..., normalize=True)` (sigmoïde, sortie dans [0,1]) sur
le chemin BGE-Reranker réel — mais son repli heuristique (pas de GPU/torch)
retourne un score composite non borné (score de retrieval brut + bonus). Les
scores sont donc bornés défensivement ici (`min(1.0, max(0.0, s))`), mais le
signal est moins fiable quand le repli heuristique est actif.
"""

from typing import Literal

from pydantic import BaseModel, Field

from src.reliability.sufficiency import SufficiencyVerdict

_SUFFICIENCY_SCORE_BY_VERDICT = {"sufficient": 1.0, "partial": 0.5, "insufficient": 0.0}


class AbstentionResult(BaseModel):
    zone: Literal["correct", "ambiguous", "incorrect"]
    combined_score: float = Field(ge=0.0, le=1.0)
    action: Literal["answer", "answer_with_caveat", "abstain"]


class AbstentionGate:
    """Combine les 4 sources de score de la spec §8 en une zone Correct/
    Ambiguous/Incorrect et l'action correspondante."""

    def __init__(self, threshold_low: float = 0.4, threshold_high: float = 0.7):
        if not 0.0 <= threshold_low < threshold_high <= 1.0:
            raise ValueError("threshold_low must be < threshold_high, both in [0.0, 1.0]")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def evaluate(
        self,
        reranker_scores: list[float],
        generation_confidence: float,
        sufficiency_verdict: SufficiencyVerdict,
        planner_confidence: Literal["low"] | None = None,
    ) -> AbstentionResult:
        """
        Args:
            reranker_scores: rerank_score des chunks retenus (spec: "score
                reranker moyen").
            generation_confidence: GenerationResult.confidence (auto-évalué
                par le LLM, spec: "score de support du LLM, Self-RAG style").
            sufficiency_verdict: sortie de SufficiencyChecker.check() (spec:
                "score de suffisance").
            planner_confidence: seul signal actuellement exposé par
                HybridRetriever est un flag "low" (intent ambiguous) ou
                l'absence de flag — pas le high/medium/low complet envisagé
                par la spec (§7 planner). Traité ici comme une pénalité
                plutôt qu'une composante pondérée à part entière, faute de
                granularité disponible.
        """
        reranker_avg = (
            sum(min(1.0, max(0.0, s)) for s in reranker_scores) / len(reranker_scores)
            if reranker_scores
            else 0.0
        )
        sufficiency_component = _SUFFICIENCY_SCORE_BY_VERDICT[sufficiency_verdict.verdict]

        combined = (
            0.3 * reranker_avg
            + 0.3 * min(1.0, max(0.0, generation_confidence))
            + 0.3 * sufficiency_component
            + 0.1 * sufficiency_verdict.confidence
        )
        if planner_confidence == "low":
            combined *= 0.85
        combined = min(1.0, max(0.0, combined))

        if combined > self.threshold_high:
            zone: Literal["correct", "ambiguous", "incorrect"] = "correct"
            action: Literal["answer", "answer_with_caveat", "abstain"] = "answer"
        elif combined >= self.threshold_low:
            zone, action = "ambiguous", "answer_with_caveat"
        else:
            zone, action = "incorrect", "abstain"

        return AbstentionResult(zone=zone, combined_score=round(combined, 3), action=action)
