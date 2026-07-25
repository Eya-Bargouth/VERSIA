"""Tests unitaires pour l'abstraction LLM."""

import pytest

from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig, LLMMessage, LLMResponse, LLMUsage
from src.llm.providers.ollama_client import OllamaClient
from src.llm.providers.vllm_client import VLLMClient

pytestmark = pytest.mark.phase1


class TestLLMFactory:
    def test_register_and_create_ollama(self):
        client = LLMFactory.create(LLMConfig(provider="ollama", model="mistral:7b"))
        assert isinstance(client, OllamaClient)

    def test_register_and_create_vllm(self):
        client = LLMFactory.create(LLMConfig(provider="vllm", model="llama3"))
        assert isinstance(client, VLLMClient)

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            LLMFactory.create(LLMConfig(provider="openai", model="gpt-4"))

    def test_list_providers(self):
        providers = LLMFactory.list_providers()
        assert "ollama" in providers
        assert "vllm" in providers


class TestLLMModels:
    def test_llm_message_serialization(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_llm_response_default(self):
        resp = LLMResponse(content="test", model="mistral")
        assert resp.usage.total_tokens == 0
        assert resp.finish_reason is None

    def test_llm_config_defaults(self):
        cfg = LLMConfig(provider="ollama", model="mistral")
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 2048
        assert cfg.base_url == "http://localhost:11434"


class TestMockLLMClient:
    def test_mock_client_complete(self, mock_llm_client):
        client = mock_llm_client()
        resp = client.complete(
            [LLMMessage(role="user", content="test")],
            LLMConfig(provider="ollama", model="mistral")
        )
        assert isinstance(resp, LLMResponse)
        assert resp.usage.total_tokens == 20

    def test_mock_client_validate(self, mock_llm_client):
        client = mock_llm_client()
        assert client.validate_config(LLMConfig(provider="ollama", model="x")) is True