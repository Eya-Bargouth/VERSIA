"""Interface abstraite et modèles pour les clients LLM."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Message dans une conversation LLM."""

    role: str
    content: str


class LLMUsage(BaseModel):
    """Compteurs de tokens."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Réponse structurée d'un LLM."""

    content: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str | None = None
    raw_response: dict = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration d'un appel LLM."""

    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str | None = None
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    fallback_provider: str | None = None
    fallback_model: str | None = None
    # JSON schema (ex. SomeModel.model_json_schema()) contraignant la sortie du
    # LLM. Chaque provider l'adapte à son propre format de payload (voir
    # OllamaClient/VLLMClient) — l'appelant ne manipule que ce schéma
    # générique, jamais le format spécifique d'un provider.
    response_format: dict | None = None
    # Nombre de couches déchargées sur GPU (Ollama uniquement). 0 force le
    # modèle entièrement sur CPU — utilisé pour le juge RAGAS (Phase 5) afin
    # d'éviter toute contention VRAM avec le pipeline évalué. None laisse
    # Ollama décider (comportement par défaut, inchangé).
    num_gpu: int | None = None
    # Pénalise la réémission de tokens déjà produits — voir
    # Settings.llm_repeat_penalty. None laisse le provider utiliser sa
    # propre valeur par défaut.
    repeat_penalty: float | None = None


def normalize_llm_scale(value):
    """Ramène une valeur numérique auto-évaluée par un LLM (confidence,
    sufficiency_score, ...) sur l'échelle 0.0-1.0 attendue partout dans le
    code (RawGenerationOutput, _RawSufficiencyOutput), même quand le modèle
    répond sur une échelle 0-100 malgré la consigne explicite du prompt.

    Mesuré en conditions réelles avec qwen2.5:3b-instruct : confidence=100
    et sufficiency_score=97 renvoyés malgré un prompt disant explicitement
    "jamais un pourcentage" — la consigne de prompt seule ne suffit pas à
    garantir le respect de l'échelle. Sans cette normalisation, Pydantic
    rejette la valeur (`le=1.0`) et la réponse entière est jetée par
    l'appelant (voir Generator._parse / SufficiencyChecker._check_llm),
    perdant une réponse par ailleurs valide pour un seul champ mal calibré.

    Ne couvre que le cas mesuré (échelle 0-100) : une valeur > 1 et <= 100
    est divisée par 100 ; au-delà, elle est plafonnée à 1.0 plutôt que
    rejetée — une valeur non numérique est retournée telle quelle pour que
    la validation Pydantic normale produise son erreur habituelle."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1:
        return min(value / 100, 1.0) if value <= 100 else 1.0
    return value


class BaseLLMClient(ABC):
    """Client LLM abstrait — toutes les implémentations doivent hériter de cette classe."""

    @abstractmethod
    def complete(self, messages: list[LLMMessage], config: LLMConfig) -> LLMResponse:
        """Appel synchrone complet."""
        ...

    @abstractmethod
    async def complete_stream(self, messages: list[LLMMessage], config: LLMConfig) -> AsyncIterator[str]:
        """Appel asynchrone en streaming."""
        ...

    @abstractmethod
    def validate_config(self, config: LLMConfig) -> bool:
        """Vérifie que la configuration est valide pour ce provider."""
        ...

    def _build_messages_payload(self, messages: list[LLMMessage]) -> list[dict]:
        """Sérialise les messages pour les APIs HTTP."""
        return [{"role": m.role, "content": m.content} for m in messages]