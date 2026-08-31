"""Cache Redis des réponses POST /query — hors périmètre spec v2.1, ajout
demandé séparément (interface de vérification /ui). Évite de régénérer une
réponse identique (retrieval + génération + juges, plusieurs dizaines de
secondes à plusieurs minutes sur qwen2.5:3b-instruct) quand la même question
est reposée.

Clé = hash(question normalisée + top_k) ; valeur = PipelineResult sérialisé
en JSON ; TTL optionnel (Settings.redis_cache_ttl_seconds, None par défaut =
pas d'expiration). L'invalidation principale est l'appel explicite à clear()
après une ré-ingestion réussie (voir routes/ingest.py) — sans TTL, c'est le
SEUL mécanisme d'invalidation, une entrée reste donc en cache indéfiniment
tant qu'aucune ré-ingestion n'a lieu.

Un Redis injoignable ou une erreur de (dé)sérialisation ne doit jamais faire
échouer POST /query — dégrade silencieusement vers un cache miss (même
principe que sparse_search_failed dans HybridRetriever) : mieux vaut
régénérer la réponse que de casser l'endpoint pour un problème de cache.
"""

import hashlib

import redis
import structlog

from src.pipeline import PipelineResult

logger = structlog.get_logger(__name__)

_KEY_PREFIX = "trade:query:"


class QueryCache:
    def __init__(self, host: str, port: int, ttl_seconds: int | None):
        self.ttl_seconds = ttl_seconds
        self._client: redis.Redis | None = None
        try:
            client = redis.Redis(host=host, port=port, socket_connect_timeout=2.0, socket_timeout=2.0)
            client.ping()
            self._client = client
        except Exception as exc:
            logger.warning("redis_unavailable_cache_disabled", host=host, port=port, error=str(exc))

    @staticmethod
    def _key(question: str, top_k: int) -> str:
        # Normalisation minimale (espaces, casse) — un cache exact-match,
        # pas de déduplication sémantique (question reformulée différemment
        # = cache miss, assumé).
        normalized = " ".join(question.strip().lower().split())
        digest = hashlib.sha256(f"{normalized}|{top_k}".encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    def get(self, question: str, top_k: int) -> PipelineResult | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(self._key(question, top_k))
            if raw is None:
                return None
            return PipelineResult.model_validate_json(raw)
        except Exception as exc:
            logger.warning("redis_cache_read_failed", error=str(exc))
            return None

    def set(self, question: str, top_k: int, result: PipelineResult) -> None:
        if self._client is None:
            return
        try:
            self._client.set(self._key(question, top_k), result.model_dump_json(), ex=self.ttl_seconds)
        except Exception as exc:
            logger.warning("redis_cache_write_failed", error=str(exc))

    def clear(self) -> None:
        """Vide tout le cache — appelé après une ré-ingestion réussie (voir
        routes/ingest.py) : les réponses en cache peuvent référencer un
        contexte devenu périmé (chunks modifiés/supprimés)."""
        if self._client is None:
            return
        try:
            keys = self._client.keys(f"{_KEY_PREFIX}*")
            if keys:
                self._client.delete(*keys)
        except Exception as exc:
            logger.warning("redis_cache_clear_failed", error=str(exc))
