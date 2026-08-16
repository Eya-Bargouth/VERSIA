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