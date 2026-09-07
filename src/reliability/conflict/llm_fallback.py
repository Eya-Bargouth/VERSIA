"""LLMFallbackDetector — détection de conflits textuels par prompt binaire via
le LLM local (spec §8b, méthode "llm_fallback").
"""

from typing import Literal

import structlog
from pydantic import BaseModel

from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage
from src.reliability.conflict.types import ConflictReport

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """Tu compares deux passages de documentation technique, potentiellement issus de sources différentes, pour détecter une contradiction factuelle entre eux.

Réponds uniquement avec le JSON demandé :
- `conflict` : true si les deux passages affirment des choses incompatibles sur le même sujet précis, false sinon (y compris s'ils parlent simplement de sujets différents, ou se complètent sans se contredire).
- `type` : si `conflict` est true, classifie parmi "factual" (contradiction factuelle nette), "temporal" (les deux étaient vrais à des moments différents — ex. versions différentes d'une même règle), "opinion" (divergence d'interprétation ou de recommandation plutôt qu'un fait vérifiable). `null` si `conflict` est false.
- `explanation` : courte justification citant les deux passages, obligatoire si `conflict` est true, sinon `null`."""

_TEMPERATURES = (0.2, 0.5, 0.8)


class _RawConflictOutput(BaseModel):
    conflict: bool
    type: Literal["factual", "temporal", "opinion"] | None = None
    explanation: str | None = None


class LLMFallbackDetector:
    """Détecteur de conflits textuels par échantillonnage LLM + vote majoritaire."""

    def __init__(self, llm_client: BaseLLMClient, n_samples: int = 3):
        self.llm_client = llm_client
        self.n_samples = n_samples

    def detect(self, chunk_a: dict, chunk_b: dict, config: LLMConfig) -> ConflictReport:
        votes: list[_RawConflictOutput] = []
        for i in range(self.n_samples):
            temperature = _TEMPERATURES[i % len(_TEMPERATURES)]
            try:
                votes.append(self._query_once(chunk_a, chunk_b, config, temperature))
            except Exception as exc:
                logger.warning("llm_fallback_sample_failed", error=str(exc))

        chunk_ids = [c.get("chunk_id") for c in (chunk_a, chunk_b) if c.get("chunk_id")]
        if not votes:
            return ConflictReport(conflict=False, method="llm_fallback", chunks=chunk_ids)

        conflict_votes = sum(1 for v in votes if v.conflict)
        conflict = conflict_votes > len(votes) / 2
        confidence = round(
            (conflict_votes if conflict else len(votes) - conflict_votes) / len(votes), 2
        )
        matching = [v for v in votes if v.conflict == conflict]
        explanation = next((v.explanation for v in matching if v.explanation), None)
        conflict_type = next((v.type for v in matching if v.type), None) if conflict else None

        return ConflictReport(
            conflict=conflict,
            type=conflict_type,
            confidence=confidence,
            method="llm_fallback",
            chunks=chunk_ids,
            explanation=explanation,
        )

    def _query_once(
        self, chunk_a: dict, chunk_b: dict, config: LLMConfig, temperature: float
    ) -> _RawConflictOutput:
        source_a = chunk_a.get("payload", {}).get("source_id", "?")
        source_b = chunk_b.get("payload", {}).get("source_id", "?")
        user_message = (
            f"Passage A (source: {source_a}) :\n{chunk_a.get('text', '')}\n\n"
            f"Passage B (source: {source_b}) :\n{chunk_b.get('text', '')}"
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message),
        ]
        sample_config = config.model_copy(
            update={
                "temperature": temperature,
                "response_format": _RawConflictOutput.model_json_schema(),
            }
        )
        response = self.llm_client.complete(messages, sample_config)
        return _RawConflictOutput.model_validate_json(response.content)
