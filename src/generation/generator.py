"""Orchestrateur de génération : prompt -> LLM (JSON structuré) -> GenerationResult
(spec §8 livrables)."""

import json
import re

import structlog
from pydantic import ValidationError

from src.generation.citation import enrich_citations
from src.generation.prompt_templates import CITATION_RETRY_MESSAGE, SYSTEM_PROMPT, build_user_message
from src.generation.schemas import GenerationResult, RawCitation, RawGenerationOutput
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Generator:
    """Génère une réponse structurée (answer/citations/confidence/sufficiency_score)
    à partir d'une question et des chunks retrouvés."""

    def __init__(self, llm_client: BaseLLMClient, llm_config: LLMConfig, store=None):
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.store = store

    def generate(
        self,
        question: str,
        chunks: list[dict],
        diff_explanation: str | None = None,
    ) -> GenerationResult:
        user_message = build_user_message(question, chunks, diff_explanation, store=self.store)
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message),
        ]
        # Température=0 pour diff_explanation testée puis abandonnée : à
        # température nulle, une sortie JSON cassée (troncature sur un
        # backtick lors de la citation d'un terme technique — voir audit
        # génération) se reproduit à l'identique à chaque nouvelle tentative
        # (vérifié : reproduction octet pour octet sur 2 relances), annulant
        # tout bénéfice du mécanisme de retry existant (qui redemande une
        # génération complète — voir run_one_question_with_retry). La
        # température par défaut du projet laisse au moins une chance réelle
        # qu'une relance tire un échantillonnage différent.
        config_update = {"response_format": RawGenerationOutput.model_json_schema()}
        config = self.llm_config.model_copy(update=config_update)

        response = self.llm_client.complete(messages, config)
        raw, parsed_ok = self._parse(response.content)

        # Ne relance que si le JSON était exploitable — un contenu déjà
        # inexploitable (raw_content non-JSON) ne deviendra pas citable en
        # relançant sur le même contenu cassé, seulement bruyant.
        if parsed_ok and self._violates_output_contract(raw):
            raw = self._retry_for_output_contract(messages, config, response.content, raw)

        citations = enrich_citations(raw.citations, chunks)

        return GenerationResult(
            answer=raw.answer,
            citations=citations,
            confidence=raw.confidence,
            sufficiency_score=raw.sufficiency_score,
        )

    @staticmethod
    def _violates_output_contract(raw: RawGenerationOutput) -> bool:
        """Une sortie non-abstentive (`no_answer_found=False`) doit fournir
        à la fois un texte de réponse non vide (règle implicite du schéma)
        ET au moins une citation (règle 3 du prompt) :
        - `citations` vide → affirmation non sourcée, passait `AbstentionGate`
          en zone "correct" faute de signal contradictoire (cas mesuré dans
          l'audit génération).
        - `answer` vide malgré des `citations` non vides → mesuré en
          conditions réelles sur qwen2.5:3b-instruct (69% d'un échantillon
          réel) : le modèle extrait la bonne citation mais échoue à
          synthétiser la phrase de réponse, laissant l'utilisateur final
          recevoir une réponse vide malgré une information correcte déjà
          identifiée en interne."""
        if raw.no_answer_found:
            return False
        return not raw.answer.strip() or not raw.citations

    def _retry_for_output_contract(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
        previous_content: str,
        raw: RawGenerationOutput,
    ) -> RawGenerationOutput:
        """Une seule relance avec rappel explicite du contrat de sortie. Si
        la violation persiste, dégradation explicite (confidence plafonnée)
        au lieu de laisser passer silencieusement une réponse vide ou non
        citée — le signal dégradé alimente ensuite
        AbstentionGate.generation_confidence."""
        retry_messages = [
            *messages,
            LLMMessage(role="assistant", content=previous_content),
            LLMMessage(role="user", content=CITATION_RETRY_MESSAGE),
        ]
        response = self.llm_client.complete(retry_messages, config)
        retried, _ = self._parse(response.content)

        if self._violates_output_contract(retried):
            logger.warning(
                "generation_output_contract_violated_after_retry",
                answer=retried.answer[:200],
                n_citations=len(retried.citations),
            )
            update = {"confidence": min(retried.confidence, 0.3)}
            # `answer` vide malgré des citations valides persistant après
            # relance (mesuré en conditions réelles sur qwen2.5:3b-instruct :
            # la relance ne corrige jamais ce cas précis) : synthèse
            # déterministe depuis les `claim` plutôt que de renvoyer une
            # chaîne vide à l'utilisateur final quand l'information est déjà
            # là. La confidence plafonnée ci-dessus reste le signal de
            # dégradation pour AbstentionGate ; ceci garantit seulement
            # qu'une réponse existe quand l'information est déjà identifiée.
            if not retried.answer.strip() and retried.citations:
                update["answer"] = self._synthesize_answer_from_citations(retried.citations)
            retried = retried.model_copy(update=update)

        return retried

    @staticmethod
    def _synthesize_answer_from_citations(citations: list[RawCitation]) -> str:
        """Reconstruit un texte de réponse minimal à partir des `claim` des
        citations — filet de sécurité déterministe, pas une synthèse LLM.
        Dédoublonne (le modèle répète parfois le même claim sur plusieurs
        citations, observé en conditions réelles)."""
        seen: set[str] = set()
        unique_claims = []
        for c in citations:
            claim = (c.claim or "").strip()
            if claim and claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)
        if not unique_claims:
            return ""
        return "(Réponse reconstruite automatiquement depuis les citations — non synthétisée par le modèle, à vérifier) " + " ".join(unique_claims)

    @staticmethod
    def _parse(content: str) -> tuple[RawGenerationOutput, bool]:
        """Parse la sortie JSON du LLM, avec repli si le modèle a entouré le
        JSON de balises markdown ou de texte parasite (le schema Ollama/vLLM
        contraint la structure mais pas toujours la sortie brute à 100%).

        Retourne (raw, parsed_ok) — parsed_ok=False identifie le repli
        d'erreur générique pour que l'appelant sache qu'une relance sur ce
        même contenu serait inutile (voir generate())."""
        try:
            return RawGenerationOutput.model_validate_json(content), True
        except (ValidationError, json.JSONDecodeError):
            pass

        match = _FENCE_RE.search(content)
        if match:
            try:
                return RawGenerationOutput.model_validate_json(match.group(1)), True
            except (ValidationError, json.JSONDecodeError):
                pass

        logger.error("generation_output_unparseable", raw_content=content[:500])
        return (
            RawGenerationOutput(
                answer="Erreur : la réponse du modèle n'a pas pu être interprétée.",
                citations=[],
                confidence=0.0,
                sufficiency_score=0.0,
            ),
            False,
        )
