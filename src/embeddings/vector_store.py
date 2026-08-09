"""Wrapper Qdrant avec payloads riches (hiérarchie, version, obsolescence, provenance)."""

import uuid
from uuid import UUID

import numpy as np
import structlog
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.ingestion.metadata import Chunk

logger = structlog.get_logger(__name__)


class RetrievalResult(BaseModel):
    """Résultat brut de recherche Qdrant."""

    chunk_id: UUID
    score: float
    source: str  # "dense" ou "sparse"
    payload: dict


class QdrantStore:
    """Store vectoriel Qdrant — unique base de données du système."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        """Crée la collection avec vecteurs nommés dense et sparse si absente."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info("collection_exists", name=self.collection_name)
        except Exception:
            logger.info("creating_collection", name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )

    def upsert(self, chunks: list[Chunk], embeddings: "EmbeddingBatch") -> None:
        """Écrit les chunks dans Qdrant de manière idempotente."""
        points = []
        for chunk, dense_vec, sparse_vec in zip(
            chunks, embeddings.dense, embeddings.sparse, strict=True
        ):
            # UUID5 déterministe depuis content_hash : idempotent + valide pour Qdrant local
            if chunk.content_hash:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.content_hash))
            else:
                point_id = str(chunk.chunk_id)

            payload = {
                "source_id": chunk.source_id,
                "source_path": chunk.metadata.source_path,
                "source_type": chunk.metadata.source_type,
                "text": chunk.text,
                "raw_text": chunk.raw_text,
                "hierarchy_path": chunk.hierarchy_path,
                "version_tag": chunk.version_tag,
                "version_order": chunk.version_order,
                "valid_from": chunk.valid_from.isoformat() if chunk.valid_from else None,
                "valid_until": chunk.valid_until.isoformat() if chunk.valid_until else None,
                "status": chunk.status,
                "parent_chunk_id": str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                "child_chunk_ids": [str(cid) for cid in chunk.child_chunk_ids],
                "section_title": chunk.metadata.section_title,
                "page_num": chunk.metadata.page_num,
                "line_num": chunk.metadata.line_num,
                "char_offset": chunk.metadata.char_offset,
                "chunk_id": str(chunk.chunk_id),
                "content_hash": chunk.content_hash,
            }

            sparse_vector = SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values()),
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector={"dense": dense_vec.tolist(), "sparse": sparse_vector},
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("upsert_complete", count=len(points))

    def search_dense(
        self,
        query_embedding: np.ndarray,
        filter: dict | None = None,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Recherche dense par similarité cosinus."""
        qdrant_filter = self._build_filter(filter) if filter else None
        results = self._execute_search(
            query_vector=query_embedding.tolist(),
            vector_name="dense",
            filter_obj=qdrant_filter,
            k=k,
        )
        return [
            RetrievalResult(
                chunk_id=UUID(r.payload.get("chunk_id", str(r.id))),
                score=r.score,
                source="dense",
                payload=r.payload,
            )
            for r in results
        ]

    def search_sparse(
        self,
        query_sparse: dict[int, float],
        filter: dict | None = None,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Recherche sparse (SPLADE)."""
        qdrant_filter = self._build_filter(filter) if filter else None
        sparse_vector = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )
        results = self._execute_search(
            query_vector=sparse_vector,
            vector_name="sparse",
            filter_obj=qdrant_filter,
            k=k,
        )
        return [
            RetrievalResult(
                chunk_id=UUID(r.payload.get("chunk_id", str(r.id))),
                score=r.score,
                source="sparse",
                payload=r.payload,
            )
            for r in results
        ]

    def _execute_search(self, query_vector, vector_name: str, filter_obj, k: int):
        """Wrapper défensif compatible search() et query_points()."""
        if hasattr(self.client, "search"):
            # API standard qdrant-client 1.9–1.11
            if vector_name == "sparse":
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(vector_name, query_vector),
                    query_filter=filter_obj,
                    limit=k,
                    with_payload=True,
                )
            else:
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name=vector_name, vector=query_vector),
                    query_filter=filter_obj,
                    limit=k,
                    with_payload=True,
                )

        elif hasattr(self.client, "query_points"):
            # API moderne qdrant-client 1.12+
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=filter_obj,
                limit=k,
                with_payload=True,
            )
            return resp.points

        else:
            raise RuntimeError(
                "QdrantClient has neither 'search' nor 'query_points'. "
                "Check qdrant-client version (expected >=1.9.0)."
            )

    @staticmethod
    def _build_filter(filter_dict: dict) -> Filter | None:
        """Construit un Filter Qdrant depuis un dict.

        Accepte deux formats :
        1. Plat (legacy)  : {"source_id": "stripe_specs"}
        2. Qdrant-style   : {"key": "source_id", "match": {"value": "stripe_specs"}}
        3. Combiné (must) : {"must": [{"key": ..., "match": ...}, ...]}
        """
        if not filter_dict:
            return None

        # Format 3 — already a must-dict
        if "must" in filter_dict:
            conditions = []
            for cond in filter_dict["must"]:
                if "key" in cond and "match" in cond:
                    conditions.append(
                        FieldCondition(key=cond["key"], match=MatchValue(value=cond["match"]["value"]))
                    )
            return Filter(must=conditions) if conditions else None

        # Format 2 — single Qdrant-style condition
        if "key" in filter_dict and "match" in filter_dict:
            return Filter(
                must=[
                    FieldCondition(
                        key=filter_dict["key"],
                        match=MatchValue(value=filter_dict["match"]["value"]),
                    )
                ]
            )

        # Format 1 — flat key:value
        must_conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict) and "match" in value:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value["match"]["value"]))
                )
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=must_conditions) if must_conditions else None