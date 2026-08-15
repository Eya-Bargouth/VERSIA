"""AbstentionGate — 3 seuils CRAG-like (spec §8 "Détails abstention CRAG-like").

`threshold_low=0.25` calibré (2026-08-14) sur données réelles : 49 questions
uniques passées par le vrai pipeline (Qdrant + BGE-M3 + reranker + Ollama),
39 in-corpus (`data/eval/questions_v1.jsonl`, hors catégorie "abstention",
ne devraient pas être abstenues) + 10 hors-corpus (catégorie "abstention",
même contenu que `data/eval/questions_abstention.jsonl`, abstention
attendue). Attention : `questions_v1.jsonl` contient déjà ces 10 questions
d'abstention (category="abstention") — un premier passage de calibration les
avait chargées deux fois, une fois mal étiquetées `should_abstain=False`
depuis questions_v1.jsonl et une fois correctement `True` depuis
questions_abstention.jsonl, biaisant le calcul ; corrigé (déduplication +
exclusion explicite, voir `scripts/calibrate_abstention_thresholds.py`).
Balayage hors-ligne de threshold_low sur les combined_score réels obtenus
(voir `data/eval/abstention_calibration_raw.jsonl`) : 0.25 retenu comme
compromis (8% faux-positifs d'abstention sur les questions in-corpus, 40%
d'abstention correcte sur les questions hors-corpus, 82% d'exactitude
globale) plutôt que le seuil qui maximise l'exactitude brute (0.10 →
seulement 10% d'abstention correcte, le mécanisme devient quasi inutile
car la classe in-corpus domine numériquement le calcul d'exactitude —
décision actée avec l'utilisateur, qui a tranché entre plusieurs points
d'équilibre mesurés).

`threshold_high=0.7` reste NON calibré empiriquement : la vérité terrain
disponible (should_abstain binaire) ne permet de valider que la frontière
abstain/non-abstain, pas la frontière correct/ambiguous qu'il contrôle —
calibration à refaire si un jeu de données gradué (pas seulement binaire)
devient disponible.

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

    def __init__(self, threshold_low: float = 0.25, threshold_high: float = 0.7):
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
