"""QueryPipeline — orchestrateur reliant retrieve -> sufficiency -> generate
-> conflict (déclenché seulement si le contexte est jugé assez suffisant) ->
abstain (spec §8). Point d'entrée unique que la Phase 6 (route API /query)
appellera, plutôt que de réinventer cet enchaînement dans le code FastAPI.

Décision actée avec l'utilisateur (2026-08-08) : la détection de conflits
textuels (coûteuse — plusieurs appels LLM par paire cross-source) ne se
déclenche que si SufficiencyChecker juge le contexte suffisant au-delà d'un
seuil configurable — inutile de chercher des conflits sur une question dont
la réponse va de toute façon être une abstention.
"""

from typing import Literal

import structlog
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.generation.generator import Generator
from src.generation.schemas import Citation
from src.ingestion.version_diff import Change
from src.llm.interface import LLMConfig
from src.reliability.abstention import AbstentionGate
from src.reliability.conflict.detector import ConflictDetector
from src.reliability.conflict.types import ConflictReport
from src.reliability.conflict.version_diff_engine import summarize_changes
from src.reliability.sufficiency import SufficiencyChecker, SufficiencyVerdict
from src.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)


class PipelineResult(BaseModel):
    """Réponse structurée finale (sous-ensemble de la réponse API §10)."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    zone: Literal["correct", "ambiguous", "incorrect"]
    sufficiency: SufficiencyVerdict
    conflicts: list[ConflictReport] = Field(default_factory=list)
    retrieval_metadata: dict


class QueryPipeline:
    """Enchaîne tous les composants Phase 3/4 pour répondre à une question."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        sufficiency_checker: SufficiencyChecker,
        abstention_gate: AbstentionGate,
        llm_config: LLMConfig,
        conflict_detector: ConflictDetector | None = None,
        conflict_sufficiency_threshold: float = 0.7,
    ):
        self.retriever = retriever
        self.generator = generator
        self.sufficiency_checker = sufficiency_checker
        self.abstention_gate = abstention_gate
        self.llm_config = llm_config
        self.conflict_detector = conflict_detector
        self.conflict_sufficiency_threshold = conflict_sufficiency_threshold

    def answer(self, question: str, top_k: int = 10) -> PipelineResult:
        settings = get_settings()
        retrieval = self.retriever.retrieve(
            question, top_k=top_k, k_dense=settings.retrieval_k_dense, k_sparse=settings.retrieval_k_sparse
        )
        chunks = retrieval["results"]

        diff_explanation = self._diff_explanation(retrieval)
        sufficiency = self.sufficiency_checker.check(question, chunks, self.llm_config, diff_explanation=diff_explanation)
        generation = self.generator.generate(question, chunks, diff_explanation=diff_explanation)
        conflicts = self._detect_conflicts(chunks, sufficiency, retrieval)

        reranker_scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
        decision = self.abstention_gate.evaluate(
            reranker_scores=reranker_scores,
            generation_confidence=generation.confidence,
            sufficiency_verdict=sufficiency,
            planner_confidence=retrieval.get("confidence"),
        )
        answer_text, citations = self._finalize_answer(decision, generation)

        logger.info(
            "pipeline_answer_done",
            zone=decision.zone,
            action=decision.action,
            combined_score=decision.combined_score,
            conflicts_found=len(conflicts),
        )

        return PipelineResult(
            answer=answer_text,
            citations=citations,
            confidence=decision.combined_score,
            zone=decision.zone,
            sufficiency=sufficiency,
            conflicts=conflicts,
            retrieval_metadata={
                "intent": retrieval.get("planner_intent"),
                "strategy": retrieval.get("strategy"),
                "latency_ms": retrieval.get("latency_ms"),
                "candidates": len(chunks),
            },
        )

    @staticmethod
    def _diff_explanation(retrieval: dict) -> str | None:
        """Reconstruit le texte de diff pour les questions comparatives, à
        partir du DiffReport (sérialisé en dict) déjà chargé par
        HybridRetriever — pas de nouvel accès disque ici.

        Filtre aux changements pertinents pour les chunks effectivement
        retrouvés : un DiffReport couvre TOUT le document (ex. 909
        changements pour Plaid 1.19.5-beta->1.20.6), et injecter la totalité
        dans le prompt noie le changement réellement demandé sous des
        centaines de lignes sans rapport — mesuré en conditions réelles :
        hallucinations (contenu inventé absent du contexte) et verdicts
        sufficiency erratiques (même question : "partial" puis "insufficient"
        selon les runs, le LLM ne "retrouvant" la bonne ligne qu'au hasard
        dans un texte massif). change.key suit le format
        "{hierarchy_path}|{type}#{ordinal}" (voir
        src/ingestion/version_diff.py::_generic_key) — même hierarchy_path
        que celui déjà exposé dans le payload des chunks retrouvés, donc pas
        de couplage à un format de source particulier."""
        diff_report = retrieval.get("diff_report")
        if not diff_report:
            return None
        changes = [Change(**c) for c in diff_report["changes"]]

        retrieved_paths = {
            path
            for c in retrieval.get("results", [])
            if (path := c.get("payload", {}).get("hierarchy_path"))
        }
        changes = [c for c in changes if c.key.rsplit("|", 1)[0] in retrieved_paths]

        return summarize_changes(changes, diff_report["version_from"], diff_report["version_to"])

    def _detect_conflicts(
        self, chunks: list[dict], sufficiency: SufficiencyVerdict, retrieval: dict
    ) -> list[ConflictReport]:
        """Fusionne conflits textuels (cross-source, LLM) et structurels
        (déterministe, à partir du diff précalculé déjà chargé par
        HybridRetriever pour les questions comparatives). detect_structural
        existait déjà et était testé isolément, mais n'avait aucun point
        d'appel en prod — seul detect_textual était invoqué ici.

        Le seuil de suffisance (self.conflict_sufficiency_threshold) ne gate
        que detect_textual : un appel LLM par paire de chunks cross-source,
        coûteux et inutile si la question va de toute façon aboutir à une
        abstention. detect_structural, lui, ne coûte rien (simple lecture
        d'un diff déjà précalculé sur disque) et ne dépend d'aucun jugement
        LLM — le gater derrière le même seuil masquait des conflits réels
        (mesuré : n_conflicts=0 alors qu'un diff existait bel et bien,
        uniquement parce que sufficiency.verdict valait "insufficient" pour
        cette question précise)."""
        reports: list[ConflictReport] = []

        diff_report = retrieval.get("diff_report")
        if diff_report and self.conflict_detector is not None:
            structural = self.conflict_detector.detect_structural(
                diff_report["source_id"], diff_report["version_from"], diff_report["version_to"]
            )
            if structural.conflict:
                reports.append(structural)

        if self.conflict_detector is None:
            return reports
        if sufficiency.verdict == "insufficient":
            return reports
        if sufficiency.confidence < self.conflict_sufficiency_threshold:
            return reports

        reports.extend(self.conflict_detector.detect_textual(chunks, self.llm_config))
        return reports

    @staticmethod
    def _finalize_answer(decision, generation) -> tuple[str, list[Citation]]:
        """AbstentionGate a le dernier mot sur le texte visible par
        l'utilisateur — pas seulement sur des métadonnées à côté — car les
        signaux combinés se sont montrés plus fiables que la seule
        auto-évaluation du LLM (voir docs/PHASE_4_SUMMARY.md §6)."""
        if decision.action == "abstain":
            return "Information non trouvée dans la documentation.", []
        if decision.action == "answer_with_caveat":
            return f"(Confiance modérée — à vérifier) {generation.answer}", generation.citations
        return generation.answer, generation.citations
