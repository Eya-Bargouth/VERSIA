"""ConflictDetector — orchestrateur (spec §8 livrables).

Combine :
- conflits structurels : VersionDiffConflictDetector (déterministe, spec §8a)
- conflits textuels : LLMFallbackDetector

"""

from itertools import combinations

import structlog

from src.llm.interface import BaseLLMClient, LLMConfig
from src.reliability.conflict.llm_fallback import LLMFallbackDetector
from src.reliability.conflict.types import ConflictReport
from src.reliability.conflict.version_diff_engine import VersionDiffConflictDetector

logger = structlog.get_logger(__name__)


class ConflictDetector:
    def __init__(self, llm_client: BaseLLMClient | None = None, diff_dir=None):
        self.version_diff_detector = VersionDiffConflictDetector(diff_dir=diff_dir)
        self.llm_fallback_detector = LLMFallbackDetector(llm_client) if llm_client else None

    def detect_structural(
        self,
        source_id: str,
        version_from: str,
        version_to: str,
        key: str | None = None,
        hierarchy_paths: set[str] | None = None,
    ) -> ConflictReport:
        return self.version_diff_detector.detect(source_id, version_from, version_to, key, hierarchy_paths)

    def detect_textual(self, chunks: list[dict], config: LLMConfig) -> list[ConflictReport]:
        """Compare toutes les paires de chunks de sources différentes parmi
        `chunks`. Retourne uniquement les paires en conflit (conflict=True)."""
        if self.llm_fallback_detector is None:
            logger.info("conflict_detector_no_llm_client_skipping_textual")
            return []

        reports: list[ConflictReport] = []
        for chunk_a, chunk_b in _cross_source_pairs(chunks):
            report = self.llm_fallback_detector.detect(chunk_a, chunk_b, config)
            if report.conflict:
                reports.append(report)
        return reports


def _cross_source_pairs(chunks: list[dict]):
    for chunk_a, chunk_b in combinations(chunks, 2):
        source_a = chunk_a.get("payload", {}).get("source_id")
        source_b = chunk_b.get("payload", {}).get("source_id")
        if source_a and source_b and source_a != source_b:
            yield chunk_a, chunk_b
