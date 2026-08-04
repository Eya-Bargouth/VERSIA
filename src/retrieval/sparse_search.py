"""Sparse search wrapper for Qdrant SPLADE."""

from typing import List, Optional

from src.embeddings.vector_store import QdrantStore, RetrievalResult


class SparseSearch:
    """Wrapper for sparse searches using SPLADE and QdrantStore."""

    def __init__(self, store: QdrantStore):
        self.store = store

    def search_sparse(
        self,
        query_sparse: dict,
        filter: Optional[dict] = None,
        k: int = 50
    ) -> List[RetrievalResult]:
        """Perform sparse search (SPLADE).

        Args:
            query_sparse: dict {token_id: float} from SPLADE tokenizer
            filter: Optional Qdrant filter dict
            k: number of results

        Returns:
            List of RetrievalResult objects
        """
        if query_sparse is None or not query_sparse:
            raise ValueError("query_sparse is required and must not be empty")
        
        return self.store.search_sparse(query_sparse, filter=filter, k=k)
