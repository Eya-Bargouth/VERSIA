"""QueryPlanner — Classification d'intent et sélection de stratégie."""

import re
from typing import Optional, Dict, Any

import structlog

logger = structlog.get_logger(__name__)


class QueryPlanner:
    """Simple rule-based query planning.

    plan(query: str) -> dict with keys: intent, entity (optional), filters (dict)
    """

    ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\b\s*(/[-\w@%\.:\{\}/~+]*)", re.IGNORECASE)
    PATH_RE = re.compile(r"/[-\w@%\.:\{\}/~+]+")

    def __init__(
        self,
        llm_client=None,
        source_aliases: Optional[Dict[str, str]] = None,
        version_tags: Optional[list] = None,
    ):
        """
        Args:
            llm_client: Optional BaseLLMClient used as fallback when intent is
                ambiguous.
            source_aliases: Optional {keyword_lowercase: source_id} map used to
                detect a source filter from the query text.
            version_tags: Optional list of known version tags (e.g. "v2323")
                used to detect a version filter from the query text.
                When either is not provided, it is discovered dynamically by
                walking ``settings.raw_data_dir`` (see
                ``_discover_source_index``). No corpus name or version tag is
                ever hardcoded here — everything comes from each discovered
                source's own ``source_id`` (and its optional versioning
                override, see SourceConfig).
        """
        self.llm_client = llm_client
        needs_discovery = source_aliases is None or version_tags is None
        discovered_aliases, discovered_versions = (
            self._discover_source_index() if needs_discovery else ({}, [])
        )
        self.source_aliases = source_aliases if source_aliases is not None else discovered_aliases
        self.version_tags = version_tags if version_tags is not None else discovered_versions

    @staticmethod
    def _discover_source_index():
        """Build (source_aliases, version_tags) from auto-discovered sources.

        Generic by construction: this function contains no corpus name and no
        version literal. It only reads whatever sources are discovered under
        the configured raw data directory and uses each source's own
        ``source_id`` (also split on "_" into extra alias tokens, since
        source_id is now the sole naming signal — no more query_aliases
        declared by hand) and optional ``version_order`` (from a versioning
        override, when one exists).
        """
        try:
            from src.config.settings import get_settings
            from src.ingestion.pipeline import discover_sources

            settings = get_settings()
            sources = discover_sources(settings.raw_data_dir, versioning_dir=settings.versioning_dir)
        except Exception as exc:
            logger.warning("planner_source_discovery_failed", error=str(exc))
            return {}, []

        aliases: Dict[str, str] = {}
        version_tags: list = []
        for source in sources:
            keywords = {source.source_id} | set(source.source_id.split("_"))
            for keyword in keywords:
                if keyword:
                    aliases[keyword.lower()] = source.source_id
            for tag in source.version_order or []:
                if tag and tag not in version_tags:
                    version_tags.append(tag)
        return aliases, version_tags

    def plan(self, query: str) -> Dict[str, Any]:
        """Plan retrieval strategy from query.

        Returns:
            dict with:
              - intent: "factual", "comparative", "navigational", "ambiguous"
              - entity: extracted path or None
              - filters: dict with optional keys status, source_id, version_tag, valid_from, valid_until
        """
        q = query.strip()
        lower = q.lower()

        # Comparative detection
        if any(w in lower for w in ("compare", "difference", "vs", "versus", "between", "différ")):
            intent = "comparative"
        # Navigational
        elif any(w in lower for w in ("where is", "show me", "find", "open", "locate", "où")):
            intent = "navigational"
        # Factual with explicit method + path
        elif self.ENDPOINT_RE.search(q) or self.PATH_RE.search(q):
            intent = "factual"
        else:
            intent = "ambiguous"

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

        # Source filter — driven entirely by manifest-declared aliases,
        # never by a corpus name hardcoded in src/ (see __init__/_discover_source_aliases).
        for keyword, source_id in self.source_aliases.items():
            if keyword in lower:
                filters["source_id"] = source_id
                break

        # Version filter — generic, from manifest-declared versioning.order tags.
        # A bare "v1"/"v2" mention (generic comparative phrasing) never pins a
        # specific tag; only a concrete known tag (e.g. "v2323") does.
        if not ("v1" in lower and "v2" in lower):
            for tag in self.version_tags:
                if tag and tag.lower() in lower:
                    filters["version_tag"] = tag
                    break

        # Temporal filters (valid_from/valid_until) — placeholder for now
        # These would be populated by more sophisticated parsing if needed
        filters["valid_from"] = None
        filters["valid_until"] = None

        return {"intent": intent, "entity": entity, "filters": filters}
