"""Fixtures pytest globales pour TRADE."""

import json
import os
from pathlib import Path

import pytest
import yaml

from src.config.manifest_schema import SourceManifest
from src.config.settings import Settings, get_settings
from src.llm.interface import LLMConfig, LLMMessage, LLMResponse, LLMUsage


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tests_dir(project_root) -> Path:
    return project_root / "tests"


@pytest.fixture(scope="session")
def fixtures_dir(tests_dir) -> Path:
    return tests_dir / "fixtures"


@pytest.fixture(scope="session")
def sample_openapi_path(fixtures_dir) -> Path:
    return fixtures_dir / "sample_openapi.yaml"


@pytest.fixture(scope="session")
def sample_markdown_path(fixtures_dir) -> Path:
    return fixtures_dir / "sample_markdown.md"


@pytest.fixture(scope="session")
def sample_json_path(fixtures_dir) -> Path:
    return fixtures_dir / "sample_incidents.json"


@pytest.fixture
def sample_manifest() -> SourceManifest:
    return SourceManifest(
        manifest_version="1.0.0",
        source_id="test_source",
        parser="markdown",
        scope={"include": ["*.md"]},
        chunking_policy={"semantic_unit": "paragraph"},
    )


@pytest.fixture
def settings() -> Settings:
    """Settings de test — ne valide pas les services distants."""
    return Settings(
        qdrant_host="localhost",
        qdrant_port=6333,
        llm_provider="ollama",
        llm_model="qwen2.5:3b-instruct",
        llm_base_url="http://localhost:11434",
    )


@pytest.fixture
def mock_llm_client():
    """Client LLM mock qui retourne des réponses prédéfinies."""
    from src.llm.interface import BaseLLMClient

    class MockLLMClient(BaseLLMClient):
        def __init__(self, responses: dict | None = None):
            self.responses = responses or {}

        def complete(self, messages, config):
            key = messages[-1].content if messages else ""
            content = self.responses.get(key, f"Mock response to: {key[:50]}")
            return LLMResponse(
                content=content,
                usage=LLMUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                model=config.model,
                raw_response={"mock": True},
            )

        async def complete_stream(self, messages, config):
            content = self.complete(messages, config).content
            for word in content.split():
                yield word + " "

        def validate_config(self, config):
            return True

    return MockLLMClient


@pytest.fixture
def ollama_config() -> LLMConfig:
    return LLMConfig(provider="ollama", model="qwen2.5:3b-instruct", base_url="http://localhost:11434")


@pytest.fixture
def vllm_config() -> LLMConfig:
    return LLMConfig(provider="vllm", model="meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://localhost:8000/v1")
