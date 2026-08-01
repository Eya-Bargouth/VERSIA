"""Configuration centralisée via pydantic-settings."""

from pathlib import Path
from typing import Literal

import httpx
import structlog
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class Settings(BaseSettings):
    """Paramètres de l'application TRADE."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "trade_chunks"

    # LLM
    llm_provider: Literal["ollama", "vllm"] = "ollama"
    llm_model: str = "mistral:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: int = 120
    llm_fallback_provider: str | None = None
    llm_fallback_model: str | None = None

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dense_dim: int = 1024
    embedding_dtype: Literal["fp32", "fp16"] = "fp16"
    embedding_backend: Literal["torch", "onnx"] = "torch"
    embedding_batch_size: int = 32
    embedding_device: str = "cuda"
    embedding_use_fp16: bool = True

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Chunking
    chunk_contextual_prefix_max_tokens: int = 100

    # Paths
    raw_data_dir: Path = Path("./raw")
    manifest_dir: Path = Path("./tests/fixtures/manifests")
    cache_dir: Path = Path("./data/cache")

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    @field_validator("raw_data_dir", "manifest_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    def validate_startup(self) -> None:
        """Vérifie que les services critiques sont joignables. Lève RuntimeError sinon."""
        errors = []

        # Qdrant
        try:
            resp = httpx.get(
                f"http://{self.qdrant_host}:{self.qdrant_port}/healthz",
                timeout=5.0,
            )
            if resp.status_code != 200:
                errors.append(f"Qdrant unhealthy: HTTP {resp.status_code}")
        except Exception as exc:
            errors.append(f"Qdrant unreachable at {self.qdrant_host}:{self.qdrant_port}: {exc}")

        # LLM provider
        try:
            if self.llm_provider == "ollama":
                resp = httpx.get(
                    f"{self.llm_base_url}/api/tags",
                    timeout=5.0,
                )
                if resp.status_code != 200:
                    errors.append(f"Ollama unhealthy: HTTP {resp.status_code}")
            elif self.llm_provider == "vllm":
                resp = httpx.get(
                    f"{self.llm_base_url}/health",
                    timeout=5.0,
                )
                if resp.status_code != 200:
                    errors.append(f"vLLM unhealthy: HTTP {resp.status_code}")
        except Exception as exc:
            errors.append(
                f"LLM provider '{self.llm_provider}' unreachable at {self.llm_base_url}: {exc}"
            )

        if errors:
            for err in errors:
                logger.error("startup_validation_failed", error=err)
            raise RuntimeError(
                f"Startup validation failed with {len(errors)} error(s): {'; '.join(errors)}"
            )

        logger.info("startup_validation_success")


# Instance globale (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings