"""Adapte src.ingestion.version_diff.VersionDiffEngine au format ConflictReport
unifié (spec §8a — conflits structurels, méthode déterministe, sans ML).

Ce module ne recalcule rien : il charge un DiffReport déjà persisté (voir
scripts/generate_version_diffs.py) et le traduit en verdict de conflit pour le
générateur / ConflictDetector. Toute paire de versions d'une même source
structurée qui diffère sur une clé donnée est par construction un conflit
"temporel" (le même objet a changé dans le temps) — jamais "factual" ni
"opinion", ces types-là relevant des conflits textuels inter-sources (§8b).
"""

import re
from pathlib import Path

import structlog

from src.ingestion.version_diff import Change, VersionDiffEngine
from src.reliability.conflict.types import ConflictReport

logger = structlog.get_logger(__name__)


class VersionDiffConflictDetector:
    """Détecteur de conflits structurels basé sur un diff déterministe précalculé."""

    def __init__(self, diff_engine: VersionDiffEngine | None = None, diff_dir: Path | None = None):
        self.diff_engine = diff_engine or VersionDiffEngine(diff_dir=diff_dir)

    def detect(
        self,
        source_id: str,
        version_from: str,
        version_to: str,
        key: str | None = None,
    ) -> ConflictReport:
        """Charge le diff précalculé (source_id, version_from, version_to) et
        le convertit en ConflictReport.

        Args:
            key: clé sémantique optionnelle (ex. "POST:/v1/orders") pour ne
                considérer que les changements touchant cet endpoint/paramètre
                précis. Sans `key`, agrège tous les changements du diff.

        Note: `ConflictReport.chunks` reste vide ici — le diff opère sur des
        clés sémantiques du DOM (avant chunking), pas sur des chunk_id Qdrant ;
        le détail est porté par `explanation` à la place.
        """
        report = self.diff_engine.load_diff(source_id, version_from, version_to)
        if report is None:
            logger.info(
                "version_diff_not_precomputed",
                source_id=source_id,
                version_from=version_from,
                version_to=version_to,
            )
            return ConflictReport(conflict=False, method="version_diff")

        changes = report.changes
        if key is not None:
            changes = [c for c in changes if c.key == key]

        if not changes:
            return ConflictReport(conflict=False, method="version_diff")

        return ConflictReport(
            conflict=True,
            type="temporal",
            confidence=1.0,  # déterministe — spec DiffReport.confidence="exact"
            method="version_diff",
            explanation=summarize_changes(changes, version_from, version_to),
        )


def summarize_changes(changes: list[Change], version_from: str, version_to: str) -> str:
    """Formate une liste de Change en texte lisible pour injection dans un
    prompt (spec §6 "le generator.py charge le diff JSON... et l'injecte dans
    le prompt"). Public : réutilisé par QueryPipeline pour les questions
    comparatives où le diff est déjà chargé par HybridRetriever."""
    lines = [f"Changements entre {version_from} et {version_to}:"]
    for change in changes:
        if change.change_type == "added":
            lines.append(f"- {change.key}: ajouté en {version_to}")
        elif change.change_type == "removed":
            lines.append(f"- {change.key}: supprimé depuis {version_from}")
        else:
            fields = ", ".join(
                f"{field} ({_clip(v['old'])!r} -> {_clip(v['new'])!r})"
                for field, v in (change.field_changes or {}).items()
            )
            lines.append(f"- {change.key}: modifié ({fields})" if fields else f"- {change.key}: modifié")
    return "\n".join(lines)


def _clip(value, max_len: int = 160) -> str:
    """Nettoie le HTML brut et tronque — évite d'injecter des Ko de markdown
    OpenAPI dans le prompt du générateur pour un simple changement de texte.
    `field_changes` peut aussi porter des valeurs non textuelles (bool/int
    pour `required`/`type` par ex.) — celles-ci passent telles quelles."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"
