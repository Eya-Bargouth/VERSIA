"""Tests citation.py — enrichissement RawCitation -> Citation."""

from uuid import uuid4

import pytest

from src.generation.citation import enrich_citations
from src.generation.schemas import RawCitation

pytestmark = pytest.mark.phase4


class TestEnrichCitations:
    def test_enriches_from_known_chunk_payload(self):
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
        raw = [RawCitation(chunk_id=chunk_id, text_span="symbol is required", support_level="fully_supported")]

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
            RawCitation(chunk_id=uuid4(), text_span="made up", support_level="fully_supported"),
        ]

        result = enrich_citations(raw, chunks)

        assert result == []

    def test_citation_ids_increment(self):
        id1, id2 = uuid4(), uuid4()
        chunks = [{"chunk_id": str(id1), "payload": {}}, {"chunk_id": str(id2), "payload": {}}]
        raw = [
            RawCitation(chunk_id=id1, text_span="a", support_level="fully_supported"),
            RawCitation(chunk_id=id2, text_span="b", support_level="partially_supported"),
        ]

        result = enrich_citations(raw, chunks)

        assert [c.citation_id for c in result] == ["cit_001", "cit_002"]

    def test_empty_input_returns_empty_list(self):
        assert enrich_citations([], []) == []
