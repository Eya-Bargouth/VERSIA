"""Dense search wrapper for Qdrant."""

from typing import List, Optional
import numpy as np

from src.embeddings.vector_store import QdrantStore, RetrievalResult


class DenseSearch:
    """Wrapper for dense vector searches using QdrantStore."""

    def __init__(self, store: QdrantStore):
        self.store = store

    def search_dense(
        self,
        query_embedding: np.ndarray,
        filter: Optional[dict] = None,
        k: int = 50
    ) -> List[RetrievalResult]:
        """Perform dense search.

        Args:
            query_embedding: 1D numpy array (float32), dimension 1024
            filter: Optional Qdrant filter dict
            k: number of results

        Returns:
            List of RetrievalResult objects
        """
        if query_embedding is None:
            raise ValueError("query_embedding is required for dense search")
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze()
        
        return self.store.search_dense(query_embedding, filter=filter, k=k)
