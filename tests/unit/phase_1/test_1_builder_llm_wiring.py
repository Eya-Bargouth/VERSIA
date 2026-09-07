"""Câblage builders -> LLM (settings, DOM builders, LLMFactory).

"""

import pytest

from src.config.source_config import SourceConfig
from src.config.settings import Settings
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.dom.models import NodeType
from src.llm.interface import LLMConfig, LLMMessage
from src.llm.factory import LLMFactory
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.llm.providers import vllm_client  # noqa: F401 — self-registers

pytestmark = pytest.mark.phase1


class TestPhase1Integration:
    def test_settings_do_not_crash_without_validation(self, settings):
        assert settings.qdrant_host is not None
        assert settings.llm_provider in ("ollama", "vllm")

    def test_yaml_builder_then_mock_llm(self, sample_openapi_path, mock_llm_client):
        config = SourceConfig(source_id="stripe_test")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)

        # Plus de types API_ENDPOINT dédiés (extraction OpenAPI spécifique
        # retirée en Phase 5) — tout passe par le type générique DOCUMENT.
        docs = [d for d in tree.get_nodes_by_type(NodeType.DOCUMENT) if d.id != tree.root_id]
        assert len(docs) > 0

        context = docs[0].markdown or ""
        client = mock_llm_client()
        resp = client.complete(
            [
                LLMMessage(role="system", content="Tu es un assistant technique."),
                LLMMessage(role="user", content=f"Explique cet endpoint: {context[:200]}")
            ],
            LLMConfig(provider="ollama", model="mistral"),
        )
        assert resp.content is not None
        assert len(resp.content) > 0

    def test_markdown_builder_then_mock_llm(self, sample_markdown_path, mock_llm_client):
        config = SourceConfig(source_id="owasp_test")
        tree = MarkdownBuilder().build(str(sample_markdown_path), config)

        assert len(tree.get_nodes_by_type(NodeType.HEADING)) > 0

        headings = [h.text for h in tree.get_nodes_by_type(NodeType.HEADING)]
        client = mock_llm_client({headings[0]: "Reponse mock"})
        resp = client.complete(
            [LLMMessage(role="user", content=headings[0])],
            LLMConfig(provider="ollama", model="mistral"),
        )
        assert "mock" in resp.content.lower() or "Reponse" in resp.content

    def test_llm_factory_providers_available(self):
        providers = LLMFactory.list_providers()
        assert "ollama" in providers
        assert "vllm" in providers
        assert "openai" not in providers
        assert "anthropic" not in providers
