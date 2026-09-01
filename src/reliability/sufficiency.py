"""Evidence Sufficiency Check (spec §8, §12.6) — tourne AVANT la génération,
sur le contexte brut retrouvé, pas sur la réponse du LLM (distinct du
`sufficiency_score` auto-évalué par le générateur — voir
src/generation/schemas.py::GenerationResult)."""

import re
from typing import Literal

import structlog
from pydantic import BaseModel, Field, field_validator

from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage, normalize_llm_scale

logger = structlog.get_logger(__name__)

_SUFFICIENCY_SYSTEM_PROMPT = """Tu évalues si un contexte documentaire contient assez d'information pour répondre à une question, sans répondre à la question toi-même.

Réponds uniquement avec le JSON demandé :
- `verdict` : "sufficient" si le contexte permet de répondre complètement et exactement, "partial" s'il ne répond que partiellement, "insufficient" s'il ne permet pas de répondre.
- `confidence` : nombre décimal entre 0.0 et 1.0 (jamais une échelle de 0 à 10)."""

# Mots vides FR/EN — corpus documentaire en anglais, questions parfois en français.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "in", "on",
    "and", "or", "what", "which", "how", "does", "do", "can", "with", "by", "at",
    "le", "la", "les", "un", "une", "des", "de", "du", "est", "sont", "pour",
    "quel", "quelle", "quels", "quelles", "comment", "que", "qui", "avec", "dans",
}


class SufficiencyVerdict(BaseModel):
    """Spec §12.6."""

    verdict: Literal["sufficient", "partial", "insufficient"]
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["llm_local", "entity_matching"]


class _RawSufficiencyOutput(BaseModel):
    verdict: Literal["sufficient", "partial", "insufficient"]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_scale(cls, v):
        return normalize_llm_scale(v)


class SufficiencyChecker:
    """Vérifie que le contexte retrouvé répond réellement à la question
    (spec §8 "Détails Evidence Sufficiency Check")."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.llm_client = llm_client

    def check(
        self,
        question: str,
        context: list[dict],
        config: LLMConfig | None = None,
        diff_explanation: str | None = None,
    ) -> SufficiencyVerdict:
        """
        Args:
            diff_explanation: Résumé texte d'un diff précalculé entre deux
                versions (même texte que celui injecté dans le prompt du
                générateur pour les questions comparatives, voir
                QueryPipeline._diff_explanation). Quand disponible, factorisé
                dans le jugement de suffisance : les chunks bruts seuls ne
                permettent jamais de répondre à "qu'est-ce qui a changé entre
                les versions X et Y", même quand le système dispose bien de
                l'information via ce diff — sans ce paramètre, le checker
                déclarait ces questions "insufficient" à tort (angle mort
                mesuré via Sufficiency Precision, spec §15.3).
        """
        if self.llm_client is not None and config is not None:
            try:
                return self._check_llm(question, context, config, diff_explanation)
            except Exception as exc:
                logger.warning("sufficiency_llm_check_failed", error=str(exc))

        return self._check_entity_matching(question, context, diff_explanation)

    def _check_llm(
        self, question: str, context: list[dict], config: LLMConfig, diff_explanation: str | None = None
    ) -> SufficiencyVerdict:
        context_text = "\n---\n".join(c.get("text", "") for c in context) or "(aucun contexte)"
        if diff_explanation:
            context_text += f"\n---\nDifférences détectées entre versions :\n{diff_explanation}"
        messages = [
            LLMMessage(role="system", content=_SUFFICIENCY_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"Question : {question}\n\n---\nContexte :\n{context_text}",
            ),
        ]
        llm_config = config.model_copy(
            update={"response_format": _RawSufficiencyOutput.model_json_schema()}
        )
        response = self.llm_client.complete(messages, llm_config)
        raw = _RawSufficiencyOutput.model_validate_json(response.content)
        return SufficiencyVerdict(
            verdict=raw.verdict, confidence=raw.confidence, method="llm_local"
        )

    @staticmethod
    def _check_entity_matching(
        question: str, context: list[dict], diff_explanation: str | None = None
    ) -> SufficiencyVerdict:
        """Repli déterministe (spec §8) : les mots significatifs de la
        question doivent apparaître dans hierarchy_path ou text des chunks
        (ou dans le résumé de diff, quand disponible)."""
        question_terms = {
            w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3 and w not in _STOPWORDS
        }
        if not question_terms:
            return SufficiencyVerdict(verdict="insufficient", confidence=0.0, method="entity_matching")

        haystack = " ".join(
            f"{c.get('text', '')} {c.get('payload', {}).get('hierarchy_path', '')}" for c in context
        ).lower()
        if diff_explanation:
            haystack += f" {diff_explanation}".lower()

        matched = sum(1 for term in question_terms if term in haystack)
        ratio = matched / len(question_terms)

        if ratio >= 0.6:
            verdict: Literal["sufficient", "partial", "insufficient"] = "sufficient"
        elif ratio > 0:
            verdict = "partial"
        else:
            verdict = "insufficient"

        return SufficiencyVerdict(verdict=verdict, confidence=round(ratio, 2), method="entity_matching")
