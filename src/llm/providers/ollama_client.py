"""Client LLM pour Ollama via HTTPX."""

from typing import AsyncIterator

import httpx
import structlog

from src.llm.factory import LLMFactory
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage, LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


class OllamaClient(BaseLLMClient):
    """Implémentation BaseLLMClient pour Ollama (endpoint /api/chat)."""

    def complete(self, messages: list[LLMMessage], config: LLMConfig) -> LLMResponse:
        url = f"{config.base_url}/api/chat"
        payload = {
            "model": config.model,
            "messages": self._build_messages_payload(messages),
            "stream": False,
            "keep_alive": "30m",  # avoid reloading the model between successive calls/runs
            # No explicit "think" field: Ollama rejects it outright (400) for
            # models without thinking capability (e.g. qwen2.5:7b-instruct —
            # confirmed: "does not support thinking"), and its default
            # behavior for models that DO support it (e.g. Qwen3) already
            # separates reasoning into message.thinking, keeping content
            # clean. Setting "think": False was tested and is actively worse
            # for thinking models — see docs/GUIDE_DEBUTANT_RAG.md.
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        if config.num_gpu is not None:
            payload["options"]["num_gpu"] = config.num_gpu
        if config.response_format is not None:
            # Ollama structured outputs: `format` prend directement le JSON
            # schema (contrairement à l'API OpenAI-compatible de vLLM, qui
            # attend un objet {"type": "json_schema", "json_schema": {...}}).
            payload["format"] = config.response_format

        try:
            resp = httpx.post(url, json=payload, timeout=config.timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("ollama_request_failed", error=str(exc), url=url)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        message = data.get("message", {})
        content = message.get("content", "")

        usage = LLMUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=(data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
        )

        return LLMResponse(
            content=content,
            usage=usage,
            model=config.model,
            finish_reason="stop" if not data.get("done_reason") else data.get("done_reason"),
            raw_response=data,
        )

    async def complete_stream(self, messages: list[LLMMessage], config: LLMConfig) -> AsyncIterator[str]:
        url = f"{config.base_url}/api/chat"
        payload = {
            "model": config.model,
            "messages": self._build_messages_payload(messages),
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        if config.num_gpu is not None:
            payload["options"]["num_gpu"] = config.num_gpu

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=config.timeout) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        import json
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})
                            yield msg.get("content", "")
                        except json.JSONDecodeError:
                            continue

    def validate_config(self, config: LLMConfig) -> bool:
        if config.provider != "ollama":
            return False
        try:
            resp = httpx.get(f"{config.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# Enregistrement automatique
LLMFactory.register("ollama", OllamaClient)
