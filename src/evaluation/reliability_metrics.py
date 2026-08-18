"""Métriques fiabilité (spec §15.3) : Citation Accuracy, Sufficiency
Precision — cibles du sujet de stage, distinctes des métriques façon RAGAS
(src/evaluation/ragas_eval.py) mais Citation Accuracy réutilise la même
mécanique de juge LLM local (BaseLLMClient, sortie JSON contrainte).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage


class _CitationVerdict(BaseModel):
    reason: str
    verdict: int = Field(ge=0, le=1)


_CITATION_ACCURACY_SYSTEM = (
    "Étant donné une affirmation précise et un passage cité à l'appui de "
    "cette affirmation, détermine si ce passage soutient effectivement "
    "l'affirmation. verdict=1 si le passage soutient réellement "
    "l'affirmation, verdict=0 s'il est hors-sujet ou ne la soutient pas. "
    "Réponds en JSON {reason, verdict}."
)


def _citation_field(citation, key: str, default=None):
    if isinstance(citation, dict):
        return citation.get(key, default)
    return getattr(citation, key, default)


def citation_accuracy(llm_client: BaseLLMClient, llm_config: LLMConfig, answer: str, citations: list) -> dict:
    """Spec §15.3, cible > 0.90 : % citations dont le text_span supporte
    effectivement l'affirmation qu'il est censé justifier — vérifié par un
    juge indépendant plutôt qu'auto-déclaré (le seul signal disponible
    jusqu'ici était `support_level`, produit par le générateur lui-même —
    voir le biais corrigé sur Hallucination Rate, hallucination.py).

    Le juge évalue chaque citation contre `claim` (l'affirmation précise
    qu'elle appuie, cf. Citation/RawCitation) plutôt que contre `answer` en
    entier — moins de bruit sémantique dans le jugement. `answer` sert de
    repli générique pour des citations sérialisées avant l'ajout de `claim`
    (ex. anciennes lignes de baseline_D_E_F_raw.jsonl)."""
    if not citations:
        return {"citation_accuracy": None, "n_citations": 0, "n_accurate": 0}

    config = llm_config.model_copy(update={"response_format": _CitationVerdict.model_json_schema()})
    accurate = 0
    for citation in citations:
        text_span = _citation_field(citation, "text_span")
        claim = _citation_field(citation, "claim") or answer
        prompt = f"Affirmation à vérifier :\n{claim}\n\nPassage cité :\n{text_span}"
        messages = [
            LLMMessage(role="system", content=_CITATION_ACCURACY_SYSTEM),
            LLMMessage(role="user", content=prompt),
        ]
        response = llm_client.complete(messages, config)
        result = _CitationVerdict.model_validate_json(response.content)
        accurate += result.verdict

    return {
        "citation_accuracy": accurate / len(citations),
        "n_citations": len(citations),
        "n_accurate": accurate,
    }


def sufficiency_precision(records: list[dict]) -> dict:
    """Spec §15.3, cible > 0.85 : % cas où SufficiencyChecker a répondu
    "insufficient" et où le contexte était *effectivement* insuffisant
    (vérité terrain : question hors-corpus, catégorie "abstention" —
    même méthodologie que la calibration d'AbstentionGate, voir
    scripts/calibrate_abstention_thresholds.py). C'est une mesure de
    PRÉCISION (quand le checker dit "insufficient", a-t-il raison), pas de
    rappel (combien d'insuffisances réelles sont détectées) — les deux sont
    différentes et la spec ne cible explicitement que la précision.

    Args:
        records: dicts avec au moins les clés "sufficiency_verdict" (str)
            et "should_abstain" (bool, vérité terrain).
    """
    flagged_insufficient = [r for r in records if r["sufficiency_verdict"] == "insufficient"]
    if not flagged_insufficient:
        return {"sufficiency_precision": None, "n_flagged_insufficient": 0, "n_correct": 0}

    correct = sum(1 for r in flagged_insufficient if r["should_abstain"])
    return {
        "sufficiency_precision": correct / len(flagged_insufficient),
        "n_flagged_insufficient": len(flagged_insufficient),
        "n_correct": correct,
    }
