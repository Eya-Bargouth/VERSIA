"""Tests citation.py — enrichissement RawCitation -> Citation."""

from uuid import uuid4

import pytest

from src.generation.citation import enrich_citations
from src.generation.schemas import RawCitation

pytestmark = pytest.mark.phase4


class TestEnrichCitations:
    def test_prefers_source_path_over_source_id(self):
        chunk_id = uuid4()
        chunks = [
            {
                "chunk_id": str(chunk_id),
                "payload": {
                    "source_id": "stripe_specs",
                    "source_path": "raw/specs-api/stripe/spec3-v2323.yaml",
                    "hierarchy_path": "paths./v1/orders.post",
                },
            }
        ]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span="symbol is required", support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert result[0].document == "raw/specs-api/stripe/spec3-v2323.yaml"

    def test_enriches_from_known_chunk_payload(self):
        """Repli sur source_id quand source_path est absent (chunks ingérés
        avant l'ajout de ce champ à ChunkMetadata)."""
        chunk_id = uuid4()
        chunks = [
            {
                "chunk_id": str(chunk_id),
                "payload": {
                    "source_id": "stripe_specs",
                    "hierarchy_path": "paths./v1/orders.post",
                    "page_num": None,
                    "line_num": 42,
                },
            }
        ]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span="symbol is required", support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert len(result) == 1
        assert result[0].citation_id == "cit_001"
        assert result[0].document == "stripe_specs"
        assert result[0].section == "paths./v1/orders.post"
        assert result[0].line == 42
        assert result[0].support_level == "fully_supported"

    def test_drops_citation_for_hallucinated_chunk_id(self):
        real_id = uuid4()
        chunks = [{"chunk_id": str(real_id), "payload": {}}]
        raw = [
            RawCitation(chunk_id=uuid4(), claim="the claim", text_span="made up", support_level="fully_supported"),
        ]

        result = enrich_citations(raw, chunks)

        assert result == []

    def test_citation_ids_increment(self):
        id1, id2 = uuid4(), uuid4()
        chunks = [{"chunk_id": str(id1), "payload": {}}, {"chunk_id": str(id2), "payload": {}}]
        raw = [
            RawCitation(chunk_id=id1, claim="the claim", text_span="a", support_level="fully_supported"),
            RawCitation(chunk_id=id2, claim="the claim", text_span="b", support_level="partially_supported"),
        ]

        result = enrich_citations(raw, chunks)

        assert [c.citation_id for c in result] == ["cit_001", "cit_002"]

    def test_empty_input_returns_empty_list(self):
        assert enrich_citations([], []) == []


class TestGroundingFallback:
    """text_span fabriqué/haluciné (absent du texte réel du chunk) : filet
    de sécurité, retombe sur le chunk entier plutôt que de laisser passer
    une citation invérifiable (audit Citation Accuracy, spec §15.3)."""

    def test_span_absent_from_chunk_falls_back_to_full_text(self):
        chunk_id = uuid4()
        chunks = [{"chunk_id": str(chunk_id), "text": "Le paramètre symbol est requis.", "payload": {}}]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span="ceci n'apparaît nulle part dans le chunk", support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert result[0].text_span == "Le paramètre symbol est requis."

    def test_span_present_verbatim_is_kept_as_is(self):
        chunk_id = uuid4()
        chunk_text = "Le paramètre symbol est requis pour cet endpoint."
        chunks = [{"chunk_id": str(chunk_id), "text": chunk_text, "payload": {}}]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span="symbol est requis", support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert result[0].text_span == "symbol est requis"

    def test_span_present_with_different_whitespace_is_kept_as_is(self):
        """Tolère les différences d'indentation/retours à la ligne (le LLM
        reproduit rarement le JSON source caractère pour caractère) sans
        déclencher le repli à tort."""
        chunk_id = uuid4()
        chunk_text = '"parameters": [\n    {\n      "name": "symbol",\n      "required": true\n    }\n  ]'
        chunks = [{"chunk_id": str(chunk_id), "text": chunk_text, "payload": {}}]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span='"name": "symbol", "required": true', support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert result[0].text_span == '"name": "symbol", "required": true'

    def test_missing_chunk_text_skips_grounding_check(self):
        """Chunk sans champ 'text' (fixture minimale/legacy) : pas de texte
        de référence disponible, on garde le text_span du LLM tel quel
        plutôt que de le vider silencieusement."""
        chunk_id = uuid4()
        chunks = [{"chunk_id": str(chunk_id), "payload": {}}]
        raw = [RawCitation(chunk_id=chunk_id, claim="the claim", text_span="symbol is required", support_level="fully_supported")]

        result = enrich_citations(raw, chunks)

        assert result[0].text_span == "symbol is required"
