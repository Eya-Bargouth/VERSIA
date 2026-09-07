"""QueryPipeline — orchestrateur reliant retrieve -> sufficiency -> generate
-> conflict (déclenché seulement si le contexte est jugé assez suffisant) ->
abstain (spec §8). Point d'entrée unique que la Phase 6 (route API /query)
appellera, plutôt que de réinventer cet enchaînement dans le code FastAPI.

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
        generation_chunks = self._narrow_chunks_for_generation(chunks, retrieval)
        sufficiency = self.sufficiency_checker.check(question, chunks, self.llm_config, diff_explanation=diff_explanation)
        generation = self.generator.generate(question, generation_chunks, diff_explanation=diff_explanation)
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
    def _target_hierarchy_paths(retrieval: dict) -> set[str]:
        """hierarchy_path à considérer comme pertinents pour la question
        comparative posée — utilisé à la fois pour filtrer le
        diff_explanation transmis au générateur et pour restreindre
        l'explication affichée par le detecteur de conflit structurel (même
        source de vérité, pour ne jamais montrer deux versions différentes
        du "changement pertinent" au générateur et à l'UI).

        Priorité au(x) nœud(s) Change trouvé(s) par recherche sémantique
        (retrieval["change_nodes"], classés par pertinence — voir
        HybridRetriever._search_relevant_changes) : ne garder QUE le meilleur
        hit restreint au seul changement demandé, plutôt que "tout
        changement dont le hierarchy_path correspond à N'IMPORTE LEQUEL des
        10 chunks retrouvés" (repli ci-dessous) — cette dernière version
        laissait passer plusieurs endpoints sans rapport partageant un nom
        de champ (ex. "consent_expiration_time" apparaît dans une dizaine de
        schémas Plaid distincts), mesurée comme cause de dérive du
        générateur vers des endpoints non demandés.

        Repli sur les hierarchy_path des chunks retrouvés si aucun nœud
        Change n'a été trouvé (index absent pour cette source, ou aucun
        changement substantiel indexé) — dégrade gracieusement plutôt que de
        ne rien filtrer du tout."""
        change_nodes = retrieval.get("change_nodes")
        if change_nodes:
            return {change_nodes[0].get("hierarchy_path")}
        return {
            path
            for c in retrieval.get("results", [])
            if (path := c.get("payload", {}).get("hierarchy_path"))
        }

    @staticmethod
    def _narrow_chunks_for_generation(chunks: list[dict], retrieval: dict) -> list[dict]:
        """Pour les questions comparatives, restreint le contexte transmis au
        générateur (pas au sufficiency checker, ni à la détection de
        conflit, qui continuent de recevoir le pool complet) aux seuls
        chunks du hierarchy_path ciblé (voir _target_hierarchy_paths),
        plutôt que le top-10 mélangé (contenu du changement demandé + N
        endpoints sans rapport partageant un score de similarité proche).

        Mesuré en conditions réelles (qwen2.5:3b ET gemma3:4b, priorités 1 et
        2 déjà en place) : diff_explanation est bien resserré sur le seul
        changement demandé, mais le générateur reçoit quand même les 10
        chunks du pool et part décrire plusieurs endpoints non demandés —
        réponse verbeuse qui épuise le budget de génération avant de clore
        le JSON (EOF juste après "sufficiency_score":, sans valeur). Ce
        resserrement cible directement cette dérive, à la source.

        Repli sur le pool complet si le filtre ne garde aucun chunk (ex.
        target_paths issu du fallback "tous les chunks retrouvés" alors
        qu'aucun ne matche réellement, cas qui ne devrait pas arriver mais
        mieux vaut risquer le pool complet qu'un contexte vide)."""
        if not retrieval.get("diff_report"):
            return chunks
        target_paths = QueryPipeline._target_hierarchy_paths(retrieval)
        narrowed = [c for c in chunks if c.get("payload", {}).get("hierarchy_path") in target_paths]
        return narrowed or chunks

    @staticmethod
    def _diff_explanation(retrieval: dict) -> str | None:
        """Reconstruit le texte de diff pour les questions comparatives, à
        partir du DiffReport (sérialisé en dict) déjà chargé par
        HybridRetriever — pas de nouvel accès disque ici.

        Filtre aux changements réellement pertinents pour la question posée
        (voir _target_hierarchy_paths) : un DiffReport couvre TOUT le
        document (ex. 909 changements pour Plaid 1.19.5-beta->1.20.6), et
        injecter la totalité dans le prompt noie le changement réellement
        demandé sous des centaines de lignes sans rapport — mesuré en
        conditions réelles : hallucinations (contenu inventé absent du
        contexte) et verdicts sufficiency erratiques. change.key suit le
        format "{hierarchy_path}|{type}#{ordinal}" (voir
        src/ingestion/version_diff.py::_generic_key) — même hierarchy_path
        que celui déjà exposé dans le payload des chunks/nœuds Change, donc
        pas de couplage à un format de source particulier."""
        diff_report = retrieval.get("diff_report")
        if not diff_report:
            return None
        changes = [Change(**c) for c in diff_report["changes"]]
        target_paths = QueryPipeline._target_hierarchy_paths(retrieval)
        changes = [c for c in changes if c.key.rsplit("|", 1)[0] in target_paths]

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
            # Même filtre que _diff_explanation (voir _target_hierarchy_paths)
            # — sans ça, `explanation` dumpait TOUS les changements du diff
            # (ex. 909 pour Plaid) dans l'UI, noyant le seul changement
            # pertinent sous des centaines sans rapport.
            structural = self.conflict_detector.detect_structural(
                diff_report["source_id"],
                diff_report["version_from"],
                diff_report["version_to"],
                hierarchy_paths=self._target_hierarchy_paths(retrieval),
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
