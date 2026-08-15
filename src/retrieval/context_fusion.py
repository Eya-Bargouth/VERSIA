"""Context fusion — labellise chunks with hierarchical context."""

from typing import List, Dict, Any, Optional


def build_labelled_contexts(
    chunks: List[Dict[str, Any]],
    version_info: Optional[Dict[str, Any]] = None,
    include_parent: bool = True,
    store: Optional[Any] = None,
) -> str:
    """Build labelled contexts for retrieved chunks.

    Contextual Retrieval (Anthropic 2024): each chunk feuille porte prefix contextuel
    résumant son contexte parent.

    Args:
        chunks: List of chunk dicts (each with 'chunk_id', 'text', 'payload' with hierarchy_path, section_title, parent_chunk_id)
        version_info: Optional version information dict
        include_parent: Whether to include parent node context (default True)
        store: Optional QdrantStore (duck-typed : tout objet avec
            `get_by_chunk_id(chunk_id) -> dict | None`). Quand fourni, un
            parent absent du lot retrouvé est explicitement récupéré par
            point lookup plutôt que d'être silencieusement omis — auparavant
            l'expansion parent ne fonctionnait que par coïncidence, quand le
            parent faisait déjà partie des chunks retournés par la recherche.

    Returns:
        Formatted string with labelled contexts
    """
    if not chunks:
        return ""

    # Index chunks by chunk_id for parent lookup
    chunk_by_id = {c.get("chunk_id"): c for c in chunks}
    _fetched_parents: Dict[str, Optional[Dict[str, Any]]] = {}

    def _resolve_parent(parent_id: str) -> Optional[Dict[str, Any]]:
        if parent_id in chunk_by_id:
            return chunk_by_id[parent_id]
        if store is None:
            return None
        if parent_id not in _fetched_parents:
            parent_payload = store.get_by_chunk_id(parent_id)
            _fetched_parents[parent_id] = (
                {"chunk_id": parent_id, "text": parent_payload.get("text", ""), "payload": parent_payload}
                if parent_payload
                else None
            )
        return _fetched_parents[parent_id]

    parts = []

    # Add version info if present
    if version_info:
        parts.append(f"## Version: {version_info.get('version_tag', 'unknown')}")
        if version_info.get('source_id'):
            parts.append(f"## Source: {version_info['source_id']}")
        parts.append("")

    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text") or chunk.get("content", "")
        payload = chunk.get("payload", {})

        # Build hierarchy prefix
        hierarchy_path = payload.get("hierarchy_path", "")
        section_title = payload.get("section_title", "")

        # Header
        parts.append(f"### [{chunk_id}] {section_title or 'Section'}")

        # Hierarchy path as context
        if hierarchy_path:
            parts.append(f"**Path:** {hierarchy_path}")

        # Include parent context if requested
        if include_parent:
            parent_id = payload.get("parent_chunk_id")
            parent_chunk = _resolve_parent(parent_id) if parent_id else None
            if parent_chunk:
                parent_payload = parent_chunk.get("payload", {})
                parent_title = parent_payload.get("section_title", "Parent Section")
                parent_text = parent_chunk.get("text", "")[:200]  # Summary (first 200 chars)
                
                parts.append(f"\n**Parent Context:** {parent_title}")
                if parent_text:
                    parts.append(f"> {parent_text}...")

        # Metadata
        if payload.get("status"):
            parts.append(f"**Status:** {payload['status']}")
        if payload.get("version_tag"):
            parts.append(f"**Version:** {payload['version_tag']}")
        if payload.get("valid_from"):
            parts.append(f"**Valid from:** {payload['valid_from']}")
        if payload.get("valid_until"):
            parts.append(f"**Valid until:** {payload['valid_until']}")

        # Main content
        parts.append(f"\n{text}\n")
        parts.append("---\n")

    return "\n".join(parts)
