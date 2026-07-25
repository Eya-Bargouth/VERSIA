"""BGE-M3 wrapper — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer


"""Wrapper BGE-M3 (dense + sparse simultanés)."""

from typing import Literal

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)


class EmbeddingBatch(BaseModel):
    """Batch d'embeddings dense + sparse."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    dense: np.ndarray                      # shape (n, 1024)
    sparse: list[dict[int, float]]         # longueur n


class BGEEmbedder:
    """Wrapper autour de BAAI/bge-m3 produisant dense + sparse (SPLADE)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dtype: Literal["fp32", "fp16"] = "fp16",
        backend: Literal["torch", "onnx"] = "torch",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        # Bascule CPU si CUDA indisponible
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda_unavailable_fallback_cpu")
            self.device = "cpu"

    def _load_model(self):
        """Lazy loading du modèle."""
        if self._model is not None:
            return
        logger.info("loading_embedding_model", model=self.model_name, device=self.device)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            logger.error("embedding_model_load_failed", error=str(exc))
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {exc}")

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Encode par lots de `batch_size`."""
        self._load_model()
        if not texts:
            return EmbeddingBatch(dense=np.zeros((0, 1024)), sparse=[])

        all_dense = []
        all_sparse = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Dense
            dense_batch = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_dense.append(dense_batch)

            # Sparse (SPLADE) — BGE-M3 expose des weights lexicaux via encode
            sparse_batch = self._model.encode(
                batch,
                convert_to_numpy=False,
                output_value="sparse",
            )
            # Normalisation des poids sparse
            for sp in sparse_batch:
                weights = {int(k): float(v) for k, v in sp.items()}
                all_sparse.append(weights)

        dense = np.vstack(all_dense) if all_dense else np.zeros((0, 1024))
        return EmbeddingBatch(dense=dense, sparse=all_sparse)

    def embed_query(self, query: str) -> tuple[np.ndarray, dict[int, float]]:
        """Encode une requête unique."""
        batch = self.embed([query])
        return batch.dense[0], batch.sparse[0]