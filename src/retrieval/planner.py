"""QueryPlanner — Classification d'intent et sélection de stratégie."""

import re
from typing import Optional, Dict, Any

import structlog
from pydantic import BaseModel, Field

from src.llm.interface import LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)


class _IntentClassification(BaseModel):
    intent: str = Field(description="one of: factual, comparative, ambiguous")


class QueryPlanner:
    """Simple rule-based query planning, avec fallback LLM optionnel quand
    les règles ne tranchent pas (intent "ambiguous").

    plan(query: str) -> dict with keys: intent, entity (optional), filters (dict)
    """

    ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\b\s*(/[-\w@%\.:\{\}/~+]*)", re.IGNORECASE)
    PATH_RE = re.compile(r"/[-\w@%\.:\{\}/~+]+")

    # Élargi au-delà de la liste fixe initiale (compare/difference/vs/versus/
    # between/différ) qui manquait des formulations courantes ("which is
    # better", "distinguish", "par rapport à", "lequel est meilleur"...).
    # \b évite les faux positifs de sous-chaîne (ex. "vs" dans un autre mot).
    COMPARATIVE_RE = re.compile(
        r"\b("
        r"compar\w*|comparaison\w*|difference\w*|differs?|différenc\w*|différ\w*"
        r"|distinguish\w*|contrast\w*"
        r"|vs\.?|versus"
        r"|better than|which (?:is|one is) (?:better|best)|lequel est (?:le )?meilleur"
        r"|meilleur(?:e)? que|par rapport (?:à|a)"
        r")\b",
        re.IGNORECASE,
    )
    COMPARATIVE_BETWEEN_RE = re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE)
    COMPARATIVE_ENTRE_ET_RE = re.compile(r"\bentre\b.+\bet\b", re.IGNORECASE)

    # "v1"/"v2" en tant que placeholders génériques ("compare v1 and v2 of the
    # API"), à distinguer d'un segment de chemin ("/v1/orders") — un "v1"
    # précédé de "/" fait partie d'une URL, jamais un tag de version générique.
    # \b après "v1"/"v2" exclut déjà les vrais tags ("v2213" etc., pas de
    # frontière de mot entre "2" et "213"). Sans le lookbehind sur "/", une
    # question mentionnant à la fois un chemin "/v1/..." (quasi systématique
    # dans ce corpus) et un vrai tag "v22xx" désactivait à tort toute
    # détection de version — trouvé en vérifiant pourquoi le filtre
    # multi-versions (voir _build_qdrant_filter) ne se déclenchait jamais sur
    # les vraies questions version_conflict du jeu de test.
    _GENERIC_V1_RE = re.compile(r"(?<!/)\bv1\b", re.IGNORECASE)
    _GENERIC_V2_RE = re.compile(r"(?<!/)\bv2\b", re.IGNORECASE)

    _INTENT_CLASSIFICATION_SYSTEM = (
        "Classe l'intention de la question suivante en une seule catégorie : "
        "'comparative' (demande explicitement de comparer deux choses, "
        "versions ou options), 'factual' (cherche un fait précis), ou "
        "'ambiguous' (aucune des précédentes ne s'applique clairement). "
        "Réponds en JSON {intent}."
    )

    def __init__(
        self,
        llm_client=None,
        llm_config: Optional[LLMConfig] = None,
        version_tags: Optional[list] = None,
    ):
        """
        Args:
            llm_client: Optional BaseLLMClient used as fallback when intent is
                ambiguous.
            llm_config: LLMConfig required alongside llm_client to actually
                perform the fallback call (provider/model/base_url). Without
                it, llm_client is accepted but never invoked — plan() stays
                purely rule-based.
            version_tags: Optional list of known version tags (e.g. "v2323")
                used to detect a version filter from the query text. When not
                provided, discovered dynamically by walking
                ``settings.raw_data_dir`` (see ``_discover_version_tags``). No
                corpus name or version tag is ever hardcoded here — everything
                comes from each discovered source's own optional versioning
                override (see SourceConfig).
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.version_tags = version_tags if version_tags is not None else self._discover_version_tags()

    @staticmethod
    def _discover_version_tags() -> list:
        """Build the known version_tags list from auto-discovered sources.

        Generic by construction: this function contains no corpus name and no
        version literal. It only reads whatever sources are discovered under
        the configured raw data directory and uses each source's optional
        ``version_order`` (from a versioning override, when one exists).
        """
        try:
            from src.config.settings import get_settings
            from src.ingestion.pipeline import discover_sources

            settings = get_settings()
            sources = discover_sources(settings.raw_data_dir, versioning_dir=settings.versioning_dir)
        except Exception as exc:
            logger.warning("planner_source_discovery_failed", error=str(exc))
            return []

        version_tags: list = []
        for source in sources:
            for tag in source.version_order or []:
                if tag and tag not in version_tags:
                    version_tags.append(tag)
        return version_tags

    def _classify_intent_llm(self, query: str) -> Optional[str]:
        """Fallback LLM pour les questions que les règles ne parviennent pas
        à classer. Ne lève jamais — une panne du juge dégrade silencieusement
        vers "ambiguous" (déjà le verdict courant à ce stade), jamais un crash."""
        try:
            config = self.llm_config.model_copy(update={"response_format": _IntentClassification.model_json_schema()})
            messages = [
                LLMMessage(role="system", content=self._INTENT_CLASSIFICATION_SYSTEM),
                LLMMessage(role="user", content=query),
            ]
            response = self.llm_client.complete(messages, config)
            result = _IntentClassification.model_validate_json(response.content)
        except Exception as exc:
            logger.warning("planner_llm_classification_failed", error=str(exc))
            return None
        if result.intent in ("factual", "comparative", "ambiguous"):
            return result.intent
        return None

    def plan(self, query: str) -> Dict[str, Any]:
        """Plan retrieval strategy from query.

        Returns:
            dict with:
              - intent: "factual", "comparative", "ambiguous"
              - entity: extracted path or None
              - filters: dict with optional keys status, version_tag, valid_from, valid_until
        """
        q = query.strip()
        lower = q.lower()

        # Comparative detection
        if (
            self.COMPARATIVE_RE.search(lower)
            or self.COMPARATIVE_BETWEEN_RE.search(lower)
            or self.COMPARATIVE_ENTRE_ET_RE.search(lower)
        ):
            intent = "comparative"
        # Factual with explicit method + path
        elif self.ENDPOINT_RE.search(q) or self.PATH_RE.search(q):
            intent = "factual"
        else:
            intent = "ambiguous"

        # Fallback LLM — seulement quand les règles ne tranchent pas, et
        # seulement si un client ET une config ont été fournis (sinon
        # plan() reste purement rule-based, comportement par défaut inchangé).
        if intent == "ambiguous" and self.llm_client is not None and self.llm_config is not None:
            llm_intent = self._classify_intent_llm(q)
            if llm_intent is not None:
                intent = llm_intent

        # Extract entity (HTTP method+path or just path)
        entity = None
        m = self.ENDPOINT_RE.search(q)
        if m:
            entity = m.group(2)
        else:
            p = self.PATH_RE.search(q)
            if p:
                entity = p.group(0)

        # Build filters from query keywords
        filters = {}

        # Status filter (default to active if not specified)
        if "active" in lower or "current" in lower:
            filters["status"] = "active"
        elif "deprecated" in lower or "obsolete" in lower:
            filters["status"] = "deprecated"
        elif "superseded" in lower:
            filters["status"] = "superseded"
        # Default to active for normal queries
        else:
            filters["status"] = "active"

        # Version filter — generic, from manifest-declared versioning.order tags.
        # A bare "v1"/"v2" mention (generic comparative phrasing) never pins a
        # specific tag; only a concrete known tag (e.g. "v2323") does.
        if not (self._GENERIC_V1_RE.search(lower) and self._GENERIC_V2_RE.search(lower)):
            matched_tags = [tag for tag in self.version_tags if tag and tag.lower() in lower]
            if intent == "comparative" and len(matched_tags) >= 2:
                # Deux versions nommées explicitement dans une question
                # comparative ("entre la version X et la version Y") : filtre
                # Qdrant à ces deux-là plutôt qu'à une seule, pour ne pas
                # laisser les autres versions non demandées polluer le top-k
                # (mesuré : c'était la cause principale des faux positifs
                # Precision@10 sur les questions version_conflict).
                filters["version_tags"] = matched_tags[:2]
            elif matched_tags:
                filters["version_tag"] = matched_tags[0]

        # Temporal filters (valid_from/valid_until) — placeholder for now
        # These would be populated by more sophisticated parsing if needed
        filters["valid_from"] = None
        filters["valid_until"] = None

        return {"intent": intent, "entity": entity, "filters": filters}
