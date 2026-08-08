"""Orchestrateur de génération : prompt -> LLM (JSON structuré) -> GenerationResult
(spec §8 livrables)."""

import json
import re

import structlog
from pydantic import ValidationError

from src.generation.citation import enrich_citations
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_message
from src.generation.schemas import GenerationResult, RawGenerationOutput
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Generator:
    """Génère une réponse structurée (answer/citations/confidence/sufficiency_score)
    à partir d'une question et des chunks retrouvés."""

    def __init__(self, llm_client: BaseLLMClient, llm_config: LLMConfig):
        self.llm_client = llm_client
        self.llm_config = llm_config

    def generate(
        self,
        question: str,
        chunks: list[dict],
        diff_explanation: str | None = None,
    ) -> GenerationResult:
        user_message = build_user_message(question, chunks, diff_explanation)
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message),
        ]
        config = self.llm_config.model_copy(
            update={"response_format": RawGenerationOutput.model_json_schema()}
        )

        response = self.llm_client.complete(messages, config)
        raw = self._parse(response.content)
        citations = enrich_citations(raw.citations, chunks)

        return GenerationResult(
            answer=raw.answer,
            citations=citations,
            confidence=raw.confidence,
            sufficiency_score=raw.sufficiency_score,
        )

    @staticmethod
    def _parse(content: str) -> RawGenerationOutput:
        """Parse la sortie JSON du LLM, avec repli si le modèle a entouré le
        JSON de balises markdown ou de texte parasite (le schema Ollama/vLLM
        contraint la structure mais pas toujours la sortie brute à 100%)."""
        try:
            return RawGenerationOutput.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError):
            pass

        match = _FENCE_RE.search(content)
        if match:
            try:
                return RawGenerationOutput.model_validate_json(match.group(1))
            except (ValidationError, json.JSONDecodeError):
                pass

        logger.error("generation_output_unparseable", raw_content=content[:500])
        return RawGenerationOutput(
            answer="Erreur : la réponse du modèle n'a pas pu être interprétée.",
            citations=[],
            confidence=0.0,
            sufficiency_score=0.0,
        )
