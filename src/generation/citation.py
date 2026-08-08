"""Enrichit les citations brutes du LLM avec les métadonnées connues du chunk
(spec §14.6). Le LLM ne fournit que `chunk_id` + `text_span` + `support_level`
(voir RawCitation) — `document`/`section`/`page`/`line` viennent du payload
Qdrant déjà retrouvé, jamais reproduits par le LLM (source d'erreur inutile
sur des données qu'on connaît déjà avec certitude).

Note connue : le payload Qdrant ne porte aucun chemin de fichier source (voir
src/embeddings/vector_store.py::upsert) — `document` utilise `source_id` en
attendant, ce qui est un identifiant de corpus (ex. "stripe_specs"), pas un
chemin de fichier au sens strict de la spec §14.6.
"""

import structlog

from src.generation.schemas import Citation, RawCitation

logger = structlog.get_logger(__name__)


def enrich_citations(raw_citations: list[RawCitation], chunks: list[dict]) -> list[Citation]:
    """Convertit les RawCitation du LLM en Citation complètes.

    Les chunk_id que le LLM n'a pas copiés depuis le contexte fourni (donc
    absents de `chunks`) sont abandonnés plutôt que de fabriquer des
    métadonnées — une citation non vérifiable ne vaut pas mieux qu'aucune
    citation.
    """
    chunk_by_id = {str(c.get("chunk_id")): c for c in chunks}
    citations: list[Citation] = []

    for raw in raw_citations:
        chunk = chunk_by_id.get(str(raw.chunk_id))
        if chunk is None:
            logger.warning("citation_dropped_unknown_chunk_id", chunk_id=str(raw.chunk_id))
            continue

        payload = chunk.get("payload", {})
        citations.append(
            Citation(
                citation_id=f"cit_{len(citations) + 1:03d}",
                chunk_id=raw.chunk_id,
                document=payload.get("source_id", "unknown"),
                section=payload.get("hierarchy_path") or payload.get("section_title"),
                page=payload.get("page_num"),
                line=payload.get("line_num"),
                text_span=raw.text_span,
                support_level=raw.support_level,
            )
        )

    return citations
