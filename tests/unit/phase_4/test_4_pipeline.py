"""Tests QueryPipeline — orchestration retrieve->sufficiency->generate->conflict->abstain."""

from uuid import uuid4

import pytest

from src.generation.schemas import Citation, GenerationResult
from src.llm.interface import LLMConfig
from src.pipeline import QueryPipeline
from src.reliability.abstention import AbstentionResult
from src.reliability.conflict.types import ConflictReport
from src.reliability.sufficiency import SufficiencyVerdict

pytestmark = pytest.mark.phase4


class _FakeRetriever:
    def __init__(self, results=None, extra=None):
        self.results = results if results is not None else [{"chunk_id": str(uuid4()), "text": "t", "payload": {}}]
        self.extra = extra or {}
        self.last_call = None

    def retrieve(self, question, top_k=10):
        self.last_call = {"question": question, "top_k": top_k}
        return {
            "results": self.results,
            "planner_intent": "factual",
            "strategy": "hybrid",
            "latency_ms": 12.3,
            **self.extra,
        }


class _FakeSufficiencyChecker:
    def __init__(self, verdict: SufficiencyVerdict):
        self.verdict = verdict
        self.called_with = None

    def check(self, question, context, config, diff_explanation=None):
        self.called_with = (question, context, config)
        self.diff_explanation_received = diff_explanation
        return self.verdict


class _FakeGenerator:
    def __init__(self, result: GenerationResult):
        self.result = result
        self.called_with = None

    def generate(self, question, chunks, diff_explanation=None):
        self.called_with = {"question": question, "chunks": chunks, "diff_explanation": diff_explanation}
        return self.result


class _FakeAbstentionGate:
    def __init__(self, result: AbstentionResult):
        self.result = result
        self.called_with = None

    def evaluate(self, **kwargs):
        self.called_with = kwargs
        return self.result


class _FakeConflictDetector:
    def __init__(self, reports=None, structural_report=None):
        self.reports = reports or []
        self.structural_report = structural_report or ConflictReport(conflict=False, method="version_diff")
        self.called = False
        self.structural_called_with = None

    def detect_textual(self, chunks, config):
        self.called = True
        return self.reports

    def detect_structural(self, source_id, version_from, version_to, key=None):
        self.structural_called_with = (source_id, version_from, version_to, key)
        return self.structural_report


def _pipeline(
    sufficiency_verdict="sufficient",
    sufficiency_confidence=0.9,
    abstention_zone="correct",
    abstention_action="answer",
    conflict_detector=None,
    conflict_sufficiency_threshold=0.7,
    retriever=None,
):
    gen_result = GenerationResult(
        answer="La réponse générée.",
        citations=[
            Citation(
                citation_id="cit_001",
                chunk_id=uuid4(),
                document="doc.yaml",
                text_span="span",
                support_level="fully_supported",
            )
        ],
        confidence=0.9,
        sufficiency_score=0.9,
    )
    return QueryPipeline(
        retriever=retriever or _FakeRetriever(),
        generator=_FakeGenerator(gen_result),
        sufficiency_checker=_FakeSufficiencyChecker(
            SufficiencyVerdict(verdict=sufficiency_verdict, confidence=sufficiency_confidence, method="llm_local")
        ),
        abstention_gate=_FakeAbstentionGate(
            AbstentionResult(zone=abstention_zone, combined_score=0.5, action=abstention_action)
        ),
        llm_config=LLMConfig(provider="ollama", model="x"),
        conflict_detector=conflict_detector,
        conflict_sufficiency_threshold=conflict_sufficiency_threshold,
    )


class TestQueryPipelineAnswerFinalization:
    def test_correct_zone_returns_generated_answer_as_is(self):
        pipeline = _pipeline(abstention_zone="correct", abstention_action="answer")
        result = pipeline.answer("Q?")
        assert result.answer == "La réponse générée."
        assert len(result.citations) == 1
        assert result.zone == "correct"

    def test_ambiguous_zone_prefixes_caveat(self):
        pipeline = _pipeline(abstention_zone="ambiguous", abstention_action="answer_with_caveat")
        result = pipeline.answer("Q?")
        assert "La réponse générée." in result.answer
        assert result.answer != "La réponse générée."
        assert len(result.citations) == 1

    def test_incorrect_zone_overrides_answer_and_drops_citations(self):
        pipeline = _pipeline(abstention_zone="incorrect", abstention_action="abstain")
        result = pipeline.answer("Q?")
        assert result.answer == "Information non trouvée dans la documentation."
        assert result.citations == []


class TestQueryPipelineConflictGating:
    def test_conflict_detector_skipped_when_none_configured(self):
        pipeline = _pipeline(conflict_detector=None)
        result = pipeline.answer("Q?")
        assert result.conflicts == []

    def test_conflict_detector_skipped_when_insufficient(self):
        detector = _FakeConflictDetector(reports=[ConflictReport(conflict=True, method="llm_fallback")])
        pipeline = _pipeline(
            sufficiency_verdict="insufficient", sufficiency_confidence=0.9, conflict_detector=detector
        )
        pipeline.answer("Q?")
        assert detector.called is False

    def test_conflict_detector_skipped_below_threshold(self):
        detector = _FakeConflictDetector(reports=[ConflictReport(conflict=True, method="llm_fallback")])
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.5,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
        )
        pipeline.answer("Q?")
        assert detector.called is False

    def test_conflict_detector_triggered_above_threshold(self):
        report = ConflictReport(conflict=True, method="llm_fallback")
        detector = _FakeConflictDetector(reports=[report])
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.9,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
        )
        result = pipeline.answer("Q?")
        assert detector.called is True
        assert result.conflicts == [report]


class TestQueryPipelineStructuralConflicts:
    """detect_structural() existait déjà et était testé isolément, mais
    n'avait jamais de point d'appel en production — seul detect_textual
    (cross-source) était invoqué. Vérifie que la question comparative
    déclenche maintenant aussi le conflit structurel déterministe, fusionné
    avec les conflits textuels."""

    _DIFF_REPORT = {
        "source_id": "stripe",
        "version_from": "legacy",
        "version_to": "v2213",
        "changes": [
            {
                "key": "GET:/v1/balance/history|param:type",
                "change_type": "modified",
                "field_changes": {"description": {"old": "a", "new": "b"}},
                "old_content_hash": "a",
                "new_content_hash": "b",
            }
        ],
        "strategy": "deterministe",
        "confidence": "exact",
    }

    def test_structural_conflict_called_and_merged_with_textual(self):
        structural = ConflictReport(conflict=True, type="temporal", method="version_diff", confidence=1.0)
        detector = _FakeConflictDetector(reports=[], structural_report=structural)
        retriever = _FakeRetriever(extra={"diff_available": True, "diff_report": self._DIFF_REPORT})
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.9,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
            retriever=retriever,
        )
        result = pipeline.answer("Q?")
        assert detector.structural_called_with == ("stripe", "legacy", "v2213", None)
        assert result.conflicts == [structural]

    def test_no_structural_conflict_reported_when_versions_actually_match(self):
        no_conflict = ConflictReport(conflict=False, method="version_diff")
        detector = _FakeConflictDetector(reports=[], structural_report=no_conflict)
        retriever = _FakeRetriever(extra={"diff_available": True, "diff_report": self._DIFF_REPORT})
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.9,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
            retriever=retriever,
        )
        result = pipeline.answer("Q?")
        assert result.conflicts == []

    def test_no_diff_report_skips_structural_check(self):
        """Question non-comparative (pas de diff_report) : pas d'appel
        detect_structural, seulement detect_textual comme avant."""
        detector = _FakeConflictDetector(reports=[])
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.9,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
        )
        pipeline.answer("Q?")
        assert detector.structural_called_with is None

    def test_textual_and_structural_both_surface_together(self):
        textual = ConflictReport(conflict=True, type="factual", method="llm_fallback")
        structural = ConflictReport(conflict=True, type="temporal", method="version_diff", confidence=1.0)
        detector = _FakeConflictDetector(reports=[textual], structural_report=structural)
        retriever = _FakeRetriever(
            results=[
                {"chunk_id": str(uuid4()), "text": "t1", "payload": {"source_id": "a"}},
                {"chunk_id": str(uuid4()), "text": "t2", "payload": {"source_id": "b"}},
            ],
            extra={"diff_available": True, "diff_report": self._DIFF_REPORT},
        )
        pipeline = _pipeline(
            sufficiency_verdict="sufficient",
            sufficiency_confidence=0.9,
            conflict_detector=detector,
            conflict_sufficiency_threshold=0.7,
            retriever=retriever,
        )
        result = pipeline.answer("Q?")
        assert result.conflicts == [textual, structural]


class TestQueryPipelineDiffExplanation:
    def test_diff_report_reconstructed_and_passed_to_generator(self):
        diff_report = {
            "source_id": "stripe_specs",
            "version_from": "legacy",
            "version_to": "v2323",
            "changes": [
                {
                    "key": "POST:/v1/orders",
                    "change_type": "modified",
                    "field_changes": {"required": {"old": False, "new": True}},
                    "old_content_hash": "a",
                    "new_content_hash": "b",
                }
            ],
            "strategy": "deterministe",
            "confidence": "exact",
        }
        retriever = _FakeRetriever(extra={"diff_available": True, "diff_report": diff_report})
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        diff_explanation = pipeline.generator.called_with["diff_explanation"]
        assert diff_explanation is not None
        assert "POST:/v1/orders" in diff_explanation

    def test_no_diff_report_passes_none(self):
        pipeline = _pipeline()
        pipeline.answer("Q?")
        assert pipeline.generator.called_with["diff_explanation"] is None

    def test_diff_explanation_also_passed_to_sufficiency_checker(self):
        """Angle mort corrigé : SufficiencyChecker jugeait "insufficient"
        les questions de conflit de version car il ne recevait jamais le
        diff précalculé, alors que le générateur (lui) l'utilise déjà pour
        répondre correctement — même texte maintenant transmis aux deux."""
        diff_report = {
            "source_id": "stripe",
            "version_from": "legacy",
            "version_to": "v2323",
            "changes": [
                {
                    "key": "POST:/v1/orders",
                    "change_type": "modified",
                    "field_changes": {"required": {"old": False, "new": True}},
                    "old_content_hash": "a",
                    "new_content_hash": "b",
                }
            ],
            "strategy": "deterministe",
            "confidence": "exact",
        }
        retriever = _FakeRetriever(extra={"diff_available": True, "diff_report": diff_report})
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        suff_diff = pipeline.sufficiency_checker.diff_explanation_received
        gen_diff = pipeline.generator.called_with["diff_explanation"]
        assert suff_diff is not None
        assert suff_diff == gen_diff
