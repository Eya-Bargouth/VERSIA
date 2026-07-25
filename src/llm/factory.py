"""Factory pour instancier les clients LLM selon la configuration."""

from src.llm.interface import BaseLLMClient, LLMConfig


class LLMFactory:
    """Registre et factory pour les providers LLM."""

    _registry: dict[str, type[BaseLLMClient]] = {}

    @classmethod
    def register(cls, name: str, client_class: type[BaseLLMClient]) -> None:
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError(f"{client_class} must inherit from BaseLLMClient")
        cls._registry[name] = client_class

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMClient:
        if config.provider not in cls._registry:
            available = ", ".join(cls.list_providers())
            raise ValueError(
                f"Provider '{config.provider}' not registered. Available: {available}"
            )
        client_class = cls._registry[config.provider]
        return client_class()

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._registry.keys())
