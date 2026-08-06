"""Client LLM pour vLLM via HTTPX (format OpenAI-compatible)."""

from typing import AsyncIterator

import httpx
import structlog

from src.llm.factory import LLMFactory
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage, LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


class VLLMClient(BaseLLMClient):
    """Implémentation BaseLLMClient pour vLLM (endpoint /v1/chat/completions)."""

    def complete(self, messages: list[LLMMessage], config: LLMConfig) -> LLMResponse:
        url = f"{config.base_url}/v1/chat/completions"
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload = {
            "model": config.model,
            "messages": self._build_messages_payload(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=config.timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("vllm_request_failed", error=str(exc), url=url)
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            usage=usage,
            model=data.get("model", config.model),
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )

    async def complete_stream(self, messages: list[LLMMessage], config: LLMConfig) -> AsyncIterator[str]:
        url = f"{config.base_url}/v1/chat/completions"
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload = {
            "model": config.model,
            "messages": self._build_messages_payload(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=config.timeout) as response:
                async for line in response.aiter_lines():
                    if line.strip().startswith("data: "):
                        import json
                        chunk_str = line[len("data: "):]
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            yield delta.get("content", "")
                        except json.JSONDecodeError:
                            continue

    def validate_config(self, config: LLMConfig) -> bool:
        if config.provider != "vllm":
            return False
        try:
            resp = httpx.get(f"{config.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# Enregistrement automatique
LLMFactory.register("vllm", VLLMClient)
