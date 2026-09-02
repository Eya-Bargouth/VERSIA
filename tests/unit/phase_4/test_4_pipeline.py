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

    def retrieve(self, question, top_k=10, k_dense=None, k_sparse=None):
        self.last_call = {"question": question, "top_k": top_k, "k_dense": k_dense, "k_sparse": k_sparse}
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

    def detect_structural(self, source_id, version_from, version_to, key=None, hierarchy_paths=None):
        self.structural_called_with = (source_id, version_from, version_to, key)
        self.structural_called_with_hierarchy_paths = hierarchy_paths
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
                claim="the claim",
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
        # Structural (gratuit, déterministe) d'abord, textual (coûteux, LLM,
        # gaté par sufficiency) ensuite — voir _detect_conflicts.
        assert result.conflicts == [structural, textual]


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
        # hierarchy_path du chunk retrouvé = préfixe (avant "|") de la clé du
        # changement — seuls les changements correspondant à un chunk
        # effectivement retrouvé sont conservés (voir _diff_explanation).
        retriever = _FakeRetriever(
            results=[{"chunk_id": str(uuid4()), "text": "t", "payload": {"hierarchy_path": "POST:/v1/orders"}}],
            extra={"diff_available": True, "diff_report": diff_report},
        )
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


class TestQueryPipelineChangeNodePriority:
    """Quand des nœuds Change ont été trouvés par recherche sémantique
    (retrieval["change_nodes"]), diff_explanation doit se restreindre au
    SEUL meilleur hit — pas à tout changement dont le hierarchy_path
    correspond à N'IMPORTE LEQUEL des chunks retrouvés. Mesuré en
    conditions réelles : plusieurs endpoints partagent un même nom de champ
    (ex. "consent_expiration_time"), et l'ancien filtre laissait passer les
    changements de TOUS ces endpoints, faisant dériver le générateur loin
    de la question posée."""

    def _diff_report(self):
        return {
            "source_id": "plaid",
            "version_from": "1.19.5-beta",
            "version_to": "1.20.6",
            "changes": [
                {
                    "key": "schemas.Item.consent_expiration_time|document#0",
                    "change_type": "modified",
                    "field_changes": {"format": {"old": None, "new": "date-time"}},
                    "old_content_hash": "a",
                    "new_content_hash": "b",
                },
                {
                    "key": "schemas.AccountsGetResponse.consent_expiration_time|document#0",
                    "change_type": "modified",
                    "field_changes": {"format": {"old": None, "new": "date-time"}},
                    "old_content_hash": "c",
                    "new_content_hash": "d",
                },
            ],
            "strategy": "deterministe",
            "confidence": "exact",
        }

    def test_only_top_change_node_kept_even_with_other_matching_chunks(self):
        retriever = _FakeRetriever(
            results=[
                {"chunk_id": str(uuid4()), "text": "t1", "payload": {"hierarchy_path": "schemas.Item.consent_expiration_time"}},
                {"chunk_id": str(uuid4()), "text": "t2", "payload": {"hierarchy_path": "schemas.AccountsGetResponse.consent_expiration_time"}},
            ],
            extra={
                "diff_available": True,
                "diff_report": self._diff_report(),
                "change_nodes": [{"hierarchy_path": "schemas.Item.consent_expiration_time"}],
            },
        )
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        diff_explanation = pipeline.generator.called_with["diff_explanation"]
        assert "schemas.Item.consent_expiration_time" in diff_explanation
        assert "schemas.AccountsGetResponse.consent_expiration_time" not in diff_explanation

    def test_falls_back_to_retrieved_chunks_when_no_change_nodes(self):
        """Sans change_nodes (index absent ou rien trouvé), repli sur le
        comportement précédent — les deux chunks retrouvés contribuent."""
        retriever = _FakeRetriever(
            results=[
                {"chunk_id": str(uuid4()), "text": "t1", "payload": {"hierarchy_path": "schemas.Item.consent_expiration_time"}},
                {"chunk_id": str(uuid4()), "text": "t2", "payload": {"hierarchy_path": "schemas.AccountsGetResponse.consent_expiration_time"}},
            ],
            extra={"diff_available": True, "diff_report": self._diff_report()},
        )
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        diff_explanation = pipeline.generator.called_with["diff_explanation"]
        assert "schemas.Item.consent_expiration_time" in diff_explanation
        assert "schemas.AccountsGetResponse.consent_expiration_time" in diff_explanation


class TestQueryPipelineNarrowedGenerationContext:
    """Priorité 3 : pour les questions comparatives, le générateur ne doit
    recevoir que les chunks du hierarchy_path ciblé (voir
    _target_hierarchy_paths), pas le pool top-10 mélangé — mesuré comme
    cause de dérive du générateur (qwen2.5:3b ET gemma3:4b) vers des
    endpoints non demandés, épuisant le budget de génération avant de
    clore le JSON. Le sufficiency checker, lui, continue de recevoir le
    pool complet (portée du changement volontairement limitée à la
    génération)."""

    def _diff_report(self):
        return {
            "source_id": "plaid",
            "version_from": "1.19.5-beta",
            "version_to": "1.20.6",
            "changes": [
                {
                    "key": "schemas.Item.consent_expiration_time|document#0",
                    "change_type": "modified",
                    "field_changes": {"format": {"old": None, "new": "date-time"}},
                    "old_content_hash": "a",
                    "new_content_hash": "b",
                }
            ],
            "strategy": "deterministe",
            "confidence": "exact",
        }

    def test_generator_receives_only_target_chunks(self):
        target_chunk = {
            "chunk_id": str(uuid4()),
            "text": "target",
            "payload": {"hierarchy_path": "schemas.Item.consent_expiration_time"},
        }
        unrelated_chunk = {
            "chunk_id": str(uuid4()),
            "text": "unrelated",
            "payload": {"hierarchy_path": "schemas.AccountsGetResponse.consent_expiration_time"},
        }
        retriever = _FakeRetriever(
            results=[target_chunk, unrelated_chunk],
            extra={
                "diff_available": True,
                "diff_report": self._diff_report(),
                "change_nodes": [{"hierarchy_path": "schemas.Item.consent_expiration_time"}],
            },
        )
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        assert pipeline.generator.called_with["chunks"] == [target_chunk]

    def test_sufficiency_checker_still_receives_full_chunk_pool(self):
        target_chunk = {
            "chunk_id": str(uuid4()),
            "text": "target",
            "payload": {"hierarchy_path": "schemas.Item.consent_expiration_time"},
        }
        unrelated_chunk = {
            "chunk_id": str(uuid4()),
            "text": "unrelated",
            "payload": {"hierarchy_path": "schemas.AccountsGetResponse.consent_expiration_time"},
        }
        retriever = _FakeRetriever(
            results=[target_chunk, unrelated_chunk],
            extra={
                "diff_available": True,
                "diff_report": self._diff_report(),
                "change_nodes": [{"hierarchy_path": "schemas.Item.consent_expiration_time"}],
            },
        )
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        _, context_passed, _ = pipeline.sufficiency_checker.called_with
        assert context_passed == [target_chunk, unrelated_chunk]

    def test_falls_back_to_full_pool_when_narrowing_would_empty_it(self):
        """Si aucun chunk retrouvé ne matche le hierarchy_path ciblé (ne
        devrait pas arriver en pratique grâce à _fetch_content_for_changes,
        mais mieux vaut garder le pool complet qu'un contexte vide)."""
        orphan_chunk = {
            "chunk_id": str(uuid4()),
            "text": "orphan",
            "payload": {"hierarchy_path": "schemas.Unrelated.other_field"},
        }
        retriever = _FakeRetriever(
            results=[orphan_chunk],
            extra={
                "diff_available": True,
                "diff_report": self._diff_report(),
                "change_nodes": [{"hierarchy_path": "schemas.Item.consent_expiration_time"}],
            },
        )
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        assert pipeline.generator.called_with["chunks"] == [orphan_chunk]

    def test_non_comparative_question_keeps_full_chunk_pool(self):
        """Sans diff_report (intent non comparatif), aucun narrowing —
        comportement inchangé pour factual/ambiguous/abstention/multi_source."""
        chunk_a = {"chunk_id": str(uuid4()), "text": "a", "payload": {"hierarchy_path": "some.path"}}
        chunk_b = {"chunk_id": str(uuid4()), "text": "b", "payload": {"hierarchy_path": "other.path"}}
        retriever = _FakeRetriever(results=[chunk_a, chunk_b])
        pipeline = _pipeline(retriever=retriever)

        pipeline.answer("Q?")

        assert pipeline.generator.called_with["chunks"] == [chunk_a, chunk_b]
