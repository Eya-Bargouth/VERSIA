"""Schémas Pydantic de la sortie du générateur (spec §8, §14.6)."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.llm.interface import normalize_llm_scale


class Citation(BaseModel):
    """Citation formalisée par segment (spec §14.6)."""

    citation_id: str
    chunk_id: UUID
    document: str
    section: str | None = None
    page: int | None = None
    line: int | None = None
    text_span: str
    support_level: Literal["fully_supported", "partially_supported", "no_support"]
    # Affirmation précise de `answer` que cette citation appuie (pas la
    # réponse entière) — permet au juge Citation Accuracy (spec §15.3) de
    # vérifier le passage contre la claim qu'il est censé soutenir plutôt que
    # contre l'ensemble de la réponse, moins bruité (voir reliability_metrics.py).
    claim: str


class RawCitation(BaseModel):
    """Citation telle que produite par le LLM — uniquement ce qu'il peut
    connaître (chunk cité, extrait, niveau de support, affirmation appuyée).
    `document`, `section`, `page`, `line` et `citation_id` sont remplis
    ensuite par ``src.generation.citation`` depuis le payload Qdrant déjà
    connu, plutôt que de faire reproduire ces métadonnées au LLM (risque
    d'erreur inutile sur des données qu'on possède déjà avec certitude)."""

    chunk_id: UUID
    text_span: str
    support_level: Literal["fully_supported", "partially_supported", "no_support"]
    claim: str


class RawGenerationOutput(BaseModel):
    """Schéma exact demandé au LLM via ``LLMConfig.response_format``."""

    answer: str
    # Déclaration structurelle explicite de l'abstention, distincte du texte
    # libre de `answer` — avant ce champ, la seule façon de savoir si une
    # réponse sans citation était une abstention légitime ("Information non
    # trouvée...") ou une affirmation factuelle non citée était de parser le
    # texte de `answer`, fragile et dépendant de la langue de la question
    # (règle 2 du prompt). Voir Generator._violates_citation_rule.
    no_answer_found: bool = False
    # max_length borne la sortie contrainte JSON (Ollama/vLLM respectent
    # maxItems du schéma) — garde-fou structurel contre une boucle de
    # répétition dégénérée du modèle (le même objet citation réémis en
    # boucle jusqu'à troncature par max_tokens, observé en conditions
    # réelles avec qwen2.5:3b-instruct) : borne le dégât quelle que soit la
    # cause, complémentaire à repeat_penalty qui s'attaque à la cause elle-même.
    citations: list[RawCitation] = Field(default_factory=list, max_length=15)
    confidence: float = Field(ge=0.0, le=1.0)
    sufficiency_score: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", "sufficiency_score", mode="before")
    @classmethod
    def _normalize_scale(cls, v):
        return normalize_llm_scale(v)


class GenerationResult(BaseModel):
    """Sortie structurée du générateur (spec §8 livrables).

    `confidence` et `sufficiency_score` sont auto-évalués par le LLM au moment
    de la génération (Self-RAG style) — un signal parmi d'autres pour
    l'AbstentionGate, distinct du SufficiencyChecker qui tourne *avant*
    génération (spec §8 "Détails Evidence Sufficiency Check").
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    sufficiency_score: float = Field(ge=0.0, le=1.0)
