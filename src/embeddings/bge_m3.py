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
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        from src.config.settings import get_settings
        settings = get_settings()

        import torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda_unavailable_fallback_cpu", message="BGE-M3 tourne sur CPU, ce sera plus lent")
            self.device = "cpu"

        # FP16 n'a d'intérêt que sur GPU
        self.use_fp16 = settings.embedding_use_fp16 if self.device == "cuda" else False
        if settings.embedding_use_fp16 and self.device == "cpu":
            logger.info("fp16_disabled_on_cpu", message="use_fp16 désactivé car le modèle tourne sur CPU")

    def _load_model(self):
        """Lazy loading du modèle."""
        if self._model is not None:
            return
        logger.info("loading_embedding_model", model=self.model_name, device=self.device, fp16=self.use_fp16)
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)
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
            
            output = self._model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            
            all_dense.append(output['dense_vecs'])
            
            # Normalisation des poids sparse
            for sp in output['lexical_weights']:
                weights = {int(k): float(v) for k, v in sp.items()}
                all_sparse.append(weights)

        dense = np.vstack(all_dense) if all_dense else np.zeros((0, 1024))
        return EmbeddingBatch(dense=dense, sparse=all_sparse)

    def embed_query(self, query: str) -> tuple[np.ndarray, dict[int, float]]:
        """Encode une requête unique."""
        batch = self.embed([query])
        return batch.dense[0], batch.sparse[0]