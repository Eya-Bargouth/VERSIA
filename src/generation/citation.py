"""Enrichit les citations brutes du LLM avec les métadonnées connues du chunk
(spec §14.6). Le LLM ne fournit que `chunk_id` + `text_span` + `support_level`
(voir RawCitation) — `document`/`section`/`page`/`line` viennent du payload
Qdrant déjà retrouvé, jamais reproduits par le LLM (source d'erreur inutile
sur des données qu'on connaît déjà avec certitude).

`document` utilise `payload["source_path"]` (chemin réel du fichier source,
ex. "raw/specs-api/stripe/spec3-v2323.yaml") avec repli sur `source_id` (juste
l'identifiant de corpus, ex. "stripe_specs") pour les chunks ingérés avant
l'ajout de ce champ à ChunkMetadata — une ré-ingestion est nécessaire pour que
`source_path` soit peuplé sur les chunks déjà dans Qdrant.
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

    `text_span` non ancré dans le texte réel du chunk (fabriqué/halluciné
    par le LLM plutôt que copié) : filet de sécurité, retombe sur le texte
    complet du chunk plutôt que de laisser passer silencieusement une
    citation invérifiable — ne corrige pas un text_span mal choisi mais
    ancré (ex. un passage réel mais non pertinent, voir Citation Accuracy
    spec §15.3), seulement une citation entièrement inventée.
    """
    chunk_by_id = {str(c.get("chunk_id")): c for c in chunks}
    citations: list[Citation] = []

    for raw in raw_citations:
        chunk = chunk_by_id.get(str(raw.chunk_id))
        if chunk is None:
            logger.warning("citation_dropped_unknown_chunk_id", chunk_id=str(raw.chunk_id))
            continue

        payload = chunk.get("payload", {})
        chunk_text = chunk.get("text", "")
        text_span = raw.text_span
        if chunk_text and text_span and _normalize_whitespace(text_span) not in _normalize_whitespace(chunk_text):
            logger.warning("citation_span_not_grounded_falling_back_to_chunk", chunk_id=str(raw.chunk_id))
            text_span = chunk_text

        citations.append(
            Citation(
                citation_id=f"cit_{len(citations) + 1:03d}",
                chunk_id=raw.chunk_id,
                document=payload.get("source_path") or payload.get("source_id", "unknown"),
                section=payload.get("hierarchy_path") or payload.get("section_title"),
                page=payload.get("page_num"),
                line=payload.get("line_num"),
                text_span=text_span,
                support_level=raw.support_level,
            )
        )

    return citations


def _normalize_whitespace(text: str) -> str:
    """Tolère les différences d'indentation/retours à la ligne entre le
    text_span reproduit par le LLM et le texte source, sans tolérer un
    contenu réellement différent."""
    return " ".join(text.split())
