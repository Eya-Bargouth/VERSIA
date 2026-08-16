"""Tests Generator — orchestration prompt -> LLM -> GenerationResult."""

from uuid import uuid4

import pytest

from src.generation.generator import Generator
from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage

pytestmark = pytest.mark.phase4


class _FakeLLMClient(BaseLLMClient):
    """Client factice retournant un contenu fixe, capture le dernier appel."""

    def __init__(self, content: str):
        self.content = content
        self.last_messages = None
        self.last_config = None

    def complete(self, messages, config):
        self.last_messages = messages
        self.last_config = config
        return LLMResponse(content=self.content, usage=LLMUsage(), model=config.model)

    async def complete_stream(self, messages, config):
        yield self.content

    def validate_config(self, config):
        return True


@pytest.fixture
def sample_chunk():
    chunk_id = uuid4()
    return chunk_id, {
        "chunk_id": str(chunk_id),
        "text": "symbol is a required string parameter",
        "payload": {"source_id": "stripe_specs", "hierarchy_path": "paths./v1/orders.post"},
    }


class TestGenerator:
    def test_sets_response_format_from_raw_schema(self, sample_chunk):
        chunk_id, chunk = sample_chunk
        client = _FakeLLMClient(
            f'{{"answer": "symbol is required", "citations": [], "confidence": 0.9, "sufficiency_score": 0.9}}'
        )
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        gen.generate("What parameter is required?", [chunk])

        assert client.last_config.response_format is not None
        assert "properties" in client.last_config.response_format

    def test_happy_path_enriches_citations(self, sample_chunk):
        chunk_id, chunk = sample_chunk
        content = (
            '{"answer": "symbol is required", '
            f'"citations": [{{"chunk_id": "{chunk_id}", "text_span": "symbol is a required string parameter", "support_level": "fully_supported"}}], '
            '"confidence": 0.9, "sufficiency_score": 0.95}'
        )
        client = _FakeLLMClient(content)
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        result = gen.generate("What parameter is required?", [chunk])

        assert result.answer == "symbol is required"
        assert result.confidence == 0.9
        assert len(result.citations) == 1
        assert result.citations[0].document == "stripe_specs"

    def test_json_wrapped_in_markdown_fence_is_parsed(self, sample_chunk):
        chunk_id, chunk = sample_chunk
        content = (
            "Voici la réponse :\n```json\n"
            '{"answer": "ok", "citations": [], "confidence": 0.5, "sufficiency_score": 0.5}\n```'
        )
        client = _FakeLLMClient(content)
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        result = gen.generate("Q", [chunk])

        assert result.answer == "ok"

    def test_unparseable_output_falls_back_safely(self, sample_chunk):
        chunk_id, chunk = sample_chunk
        client = _FakeLLMClient("this is not json at all")
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        result = gen.generate("Q", [chunk])

        assert result.confidence == 0.0
        assert result.sufficiency_score == 0.0
        assert result.citations == []

    def test_too_many_citations_falls_back_safely(self, sample_chunk):
        """Régression : boucle de répétition dégénérée observée en conditions
        réelles (qwen2.5:3b-instruct réémettant le même objet citation en
        boucle) — RawGenerationOutput.citations plafonné à 15 (spec schema
        max_length) ; si le JSON en dépasse quand même, dégradation propre
        plutôt qu'un crash non géré."""
        chunk_id, chunk = sample_chunk
        one_citation = f'{{"chunk_id": "{chunk_id}", "text_span": "x", "support_level": "fully_supported"}}'
        content = (
            '{"answer": "a", "citations": [' + ", ".join([one_citation] * 20) + '], '
            '"confidence": 0.9, "sufficiency_score": 0.9}'
        )
        client = _FakeLLMClient(content)
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        result = gen.generate("Q", [chunk])

        assert result.confidence == 0.0
        assert result.citations == []

    def test_question_precedes_context_in_sent_message(self, sample_chunk):
        chunk_id, chunk = sample_chunk
        client = _FakeLLMClient(
            '{"answer": "a", "citations": [], "confidence": 0.5, "sufficiency_score": 0.5}'
        )
        gen = Generator(client, LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"))

        gen.generate("MyQuestion", [chunk])

        user_msg = client.last_messages[-1].content
        assert user_msg.index("MyQuestion") < user_msg.index(chunk["text"])
