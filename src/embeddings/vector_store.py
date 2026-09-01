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
    KeywordIndexParams,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

try:
    # Retiré de qdrant-client 1.12+ (remplacé par query_points()/using=) — voir
    # _execute_search ci-dessous, qui ne prend la branche legacy .search() que
    # si cet import a réussi ET que le client expose encore .search().
    from qdrant_client.models import NamedVector
except ImportError:
    NamedVector = None

from src.dom.utils import derive_parent_path

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
        self.ensure_parent_path_index()

    def ensure_parent_path_index(self) -> None:
        """Index keyword sur `parent_path` — nécessaire pour retrouver
        efficacement les chunks frères (voir hybrid_retriever.py::
        _fetch_missing_siblings). Idempotent : ne lève pas si l'index existe
        déjà. Appelé séparément de la création de collection car une
        collection déjà en place (avant l'ajout de ce mécanisme) doit
        pouvoir recevoir l'index sans être recréée."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="parent_path",
                field_schema=KeywordIndexParams(type=PayloadSchemaType.KEYWORD),
            )
            logger.info("parent_path_index_created", collection=self.collection_name)
        except Exception as exc:
            logger.info("parent_path_index_already_exists_or_failed", error=str(exc))

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
                "parent_path": derive_parent_path(chunk.hierarchy_path),
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

        # Par lots : un seul appel HTTP avec des dizaines de milliers de
        # points (cas réel depuis le passage au chunking générique universel,
        # voir Phase 5) provoque une coupure de connexion côté client
        # (WinError 10053) — Qdrant encaisse très bien des lots de quelques
        # centaines de points, jamais un envoi monolithique.
        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
        logger.info("upsert_complete", count=len(points))

    def search_dense(
        self,
        query_embedding: np.ndarray,
        filter: dict | None = None,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Recherche dense par similarité cosinus.

        Exclut toujours les nœuds `node_type="change"` (voir
        `search_change_nodes`) — ce ne sont pas des chunks de contenu, un
        appelant qui oublierait de les filtrer explicitement ne doit jamais
        pouvoir les faire remonter dans une réponse factuelle par accident."""
        qdrant_filter = self._exclude_change_nodes(self._build_filter(filter) if filter else None)
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
        """Recherche sparse (SPLADE). Exclut les nœuds Change, voir search_dense."""
        qdrant_filter = self._exclude_change_nodes(self._build_filter(filter) if filter else None)
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

    @staticmethod
    def _exclude_change_nodes(filter_obj: Filter | None) -> Filter:
        """Ajoute `must_not node_type=change` à un filtre existant (ou en
        crée un) — appelé systématiquement par search_dense/search_sparse,
        jamais contournable par un filtre appelant qui aurait oublié cette
        exclusion (voir VersionRAG : les nœuds Change ne doivent jamais se
        mélanger au contenu normal du retrieval générique)."""
        exclusion = FieldCondition(key="node_type", match=MatchValue(value="change"))
        if filter_obj is None:
            return Filter(must_not=[exclusion])
        must_not = list(filter_obj.must_not or []) + [exclusion]
        return Filter(must=filter_obj.must, should=filter_obj.should, must_not=must_not)

    def search_change_nodes(
        self,
        query_embedding: np.ndarray,
        source_id: str,
        version_from: str,
        version_to: str,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Recherche sémantique restreinte aux nœuds Change (voir
        upsert_change_nodes) — chemin de retrieval dédié aux questions
        comparatives (VersionRAG §4.2 "semantic search over indexed
        changes"), jamais emprunté par le retrieval générique."""
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="node_type", match=MatchValue(value="change")),
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                FieldCondition(key="version_from", match=MatchValue(value=version_from)),
                FieldCondition(key="version_to", match=MatchValue(value=version_to)),
            ]
        )
        results = self._execute_search(
            query_vector=query_embedding.tolist(), vector_name="dense", filter_obj=qdrant_filter, k=k
        )
        return [
            RetrievalResult(chunk_id=UUID(r.payload.get("chunk_id", str(r.id))), score=r.score, source="change", payload=r.payload)
            for r in results
        ]

    def upsert_change_nodes(self, nodes: list[dict], embeddings: np.ndarray) -> None:
        """Indexe des nœuds Change (description sémantique d'un changement
        entre deux versions) comme des points Qdrant à part, marqués
        `node_type="change"` — jamais retournés par search_dense/
        search_sparse (voir _exclude_change_nodes), uniquement par
        search_change_nodes.

        `nodes` : liste de dicts {chunk_id, text, source_id, version_from,
        version_to, hierarchy_path, change_type} — `chunk_id` est ici une clé
        lisible (ex. "change-plaid-1.19.5-beta-1.20.6-<key>"), pas un UUID :
        dérivée en UUID5 déterministe pour l'id du point ET pour
        `payload["chunk_id"]` (RetrievalResult.chunk_id est typé UUID — voir
        search_change_nodes), la clé lisible d'origine reste disponible sous
        `payload["node_key"]` pour le débogage. `embeddings` : un vecteur
        dense par nœud, même ordre — même embedder que le contenu normal
        (BGE-M3), pour rester dans le même espace vectoriel."""
        points = []
        for node, vector in zip(nodes, embeddings):
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, node["chunk_id"]))
            payload = {**node, "node_type": "change", "node_key": node["chunk_id"], "chunk_id": point_uuid}
            points.append(
                PointStruct(
                    id=point_uuid,
                    vector={"dense": vector.tolist()},
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def _execute_search(self, query_vector, vector_name: str, filter_obj, k: int):
        """Wrapper défensif compatible search() et query_points()."""
        if hasattr(self.client, "search") and NamedVector is not None:
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

    def get_by_chunk_id(self, chunk_id: str) -> dict | None:
        """Récupère le payload d'un point par son `chunk_id` logique (champ
        de payload), pas l'id de point Qdrant interne (dérivé de
        content_hash — non prévisible depuis un `parent_chunk_id` stocké en
        payload). Utilisé pour l'expansion parent explicite en retrieval
        (context_fusion.build_labelled_contexts) quand le parent n'est pas
        déjà présent dans le lot retrouvé. Retourne None si absent."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]),
            limit=1,
            with_payload=True,
        )
        return points[0].payload if points else None

    def get_by_parent_path(
        self, parent_path: str, source_id: str | None = None, version_tag: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Récupère les payloads de tous les chunks partageant le même
        `parent_path` (voir src/dom/utils.py::derive_parent_path) — les
        éléments d'une même liste source (ex. les paramètres d'un endpoint),
        que le retrieval par similarité n'a pas forcément tous classés dans
        le même top-k.

        `hierarchy_path` (donc `parent_path`) est une position purement
        structurelle dans un document, identique d'une version à l'autre
        d'une même source versionnée (ex. le même endpoint Stripe existe
        avec le même chemin dans les 4 fichiers de version) — filtrer aussi
        sur `source_id`/`version_tag` est nécessaire pour ne récupérer que
        les frères de la MÊME version que le chunk d'ancrage, jamais un
        mélange de plusieurs versions.

        `limit` borne le coût : une liste source dépassant 50 éléments est
        un cas extrême non visé ici."""
        must = [FieldCondition(key="parent_path", match=MatchValue(value=parent_path))]
        if source_id:
            must.append(FieldCondition(key="source_id", match=MatchValue(value=source_id)))
        if version_tag:
            must.append(FieldCondition(key="version_tag", match=MatchValue(value=version_tag)))
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [p.payload for p in points]

    def get_by_hierarchy_path(
        self, hierarchy_path: str, source_id: str | None = None, version_tag: str | None = None, limit: int = 5
    ) -> list[dict]:
        """Récupère les payloads des chunks dont le `hierarchy_path` est
        EXACTEMENT celui donné (contrairement à `get_by_parent_path`, qui
        récupère les frères d'une même liste source) — utilisé pour aller
        chercher directement l'homologue d'une AUTRE version d'un même
        nœud, quand une question comparative en a retrouvé un côté mais pas
        l'autre (voir HybridRetriever._complete_version_pairs)."""
        must = [FieldCondition(key="hierarchy_path", match=MatchValue(value=hierarchy_path))]
        if source_id:
            must.append(FieldCondition(key="source_id", match=MatchValue(value=source_id)))
        if version_tag:
            must.append(FieldCondition(key="version_tag", match=MatchValue(value=version_tag)))
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [p.payload for p in points]

    @staticmethod
    def _build_match(match_dict: dict) -> MatchValue | MatchAny:
        """``{"value": x}`` → égalité exacte. ``{"any": [x, y]}`` → une valeur
        parmi plusieurs (ex. comparer deux versions nommées explicitement
        plutôt que d'en filtrer une seule) — voir QueryPlanner filters."""
        if "any" in match_dict:
            return MatchAny(any=match_dict["any"])
        return MatchValue(value=match_dict["value"])

    @staticmethod
    def _build_filter(filter_dict: dict) -> Filter | None:
        """Construit un Filter Qdrant depuis un dict.

        Accepte deux formats :
        1. Plat (legacy)  : {"source_id": "stripe_specs"}
        2. Qdrant-style   : {"key": "source_id", "match": {"value": "stripe_specs"}}
        3. Combiné (must) : {"must": [{"key": ..., "match": ...}, ...]}

        Le champ ``match`` accepte soit ``{"value": x}`` (égalité), soit
        ``{"any": [x, y, ...]}`` (une valeur parmi plusieurs).
        """
        if not filter_dict:
            return None

        # Format 3 — already a must-dict
        if "must" in filter_dict:
            conditions = []
            for cond in filter_dict["must"]:
                if "key" in cond and "match" in cond:
                    conditions.append(
                        FieldCondition(key=cond["key"], match=QdrantStore._build_match(cond["match"]))
                    )
            return Filter(must=conditions) if conditions else None

        # Format 2 — single Qdrant-style condition
        if "key" in filter_dict and "match" in filter_dict:
            return Filter(
                must=[
                    FieldCondition(
                        key=filter_dict["key"],
                        match=QdrantStore._build_match(filter_dict["match"]),
                    )
                ]
            )

        # Format 1 — flat key:value
        must_conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict) and ("match" in value or "any" in value):
                match_dict = value["match"] if "match" in value else value
                must_conditions.append(
                    FieldCondition(key=key, match=QdrantStore._build_match(match_dict))
                )
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=must_conditions) if must_conditions else None