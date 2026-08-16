"""Tests support JSON structuré (LLMConfig.response_format) — Ollama et vLLM."""

import json

import httpx
import pytest

from src.llm.interface import LLMConfig, LLMMessage
from src.llm.providers.ollama_client import OllamaClient
from src.llm.providers.vllm_client import VLLMClient

pytestmark = pytest.mark.phase4


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestOllamaResponseFormat:
    def test_response_format_sets_ollama_format_field(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse({"message": {"content": "{}"}, "done_reason": "stop"})

        monkeypatch.setattr(httpx, "post", fake_post)

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        config = LLMConfig(provider="ollama", model="qwen2.5:3b-instruct", response_format=schema)
        OllamaClient().complete([LLMMessage(role="user", content="hi")], config)

        assert captured["payload"]["format"] == schema

    def test_no_response_format_omits_format_field(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse({"message": {"content": "ok"}, "done_reason": "stop"})

        monkeypatch.setattr(httpx, "post", fake_post)

        config = LLMConfig(provider="ollama", model="qwen2.5:3b-instruct")
        OllamaClient().complete([LLMMessage(role="user", content="hi")], config)

        assert "format" not in captured["payload"]


class TestOllamaRepeatPenalty:
    """Contre les boucles de répétition dégénérées observées en conditions
    réelles (qwen2.5:3b-instruct réémettant le même objet citation en boucle
    à basse température) — voir Settings.llm_repeat_penalty."""

    def test_repeat_penalty_set_in_options_when_configured(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse({"message": {"content": "ok"}, "done_reason": "stop"})

        monkeypatch.setattr(httpx, "post", fake_post)

        config = LLMConfig(provider="ollama", model="qwen2.5:3b-instruct", repeat_penalty=1.3)
        OllamaClient().complete([LLMMessage(role="user", content="hi")], config)

        assert captured["payload"]["options"]["repeat_penalty"] == 1.3

    def test_repeat_penalty_omitted_when_not_configured(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse({"message": {"content": "ok"}, "done_reason": "stop"})

        monkeypatch.setattr(httpx, "post", fake_post)

        config = LLMConfig(provider="ollama", model="qwen2.5:3b-instruct")
        OllamaClient().complete([LLMMessage(role="user", content="hi")], config)

        assert "repeat_penalty" not in captured["payload"]["options"]


class TestVLLMResponseFormat:
    def test_response_format_sets_openai_json_schema(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse(
                {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "model": "x"}
            )

        monkeypatch.setattr(httpx, "post", fake_post)

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        config = LLMConfig(provider="vllm", model="llama3", base_url="http://localhost:8000", response_format=schema)
        VLLMClient().complete([LLMMessage(role="user", content="hi")], config)

        assert captured["payload"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
