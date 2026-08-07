"""Version diff engine — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer


"""VersionDiffEngine — diff déterministe sur DOM trees sans graphe."""

import json
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel

from src.dom.models import DocumentTree, NodeType

logger = structlog.get_logger(__name__)

_DIFF_DIR = Path("data/diffs")


class Change(BaseModel):
    key: str
    change_type: Literal["added", "removed", "modified"]
    field_changes: dict[str, dict[str, Any]] | None = None
    old_content_hash: str | None = None
    new_content_hash: str | None = None


class DiffReport(BaseModel):
    source_id: str
    version_from: str
    version_to: str
    changes: list[Change]
    strategy: Literal["deterministe"] = "deterministe"
    confidence: Literal["exact"] = "exact"


class VersionDiffEngine:
    """Moteur de diff déterministe entre deux versions d'une source structurée."""

    def __init__(self, diff_dir: Path | None = None):
        self.diff_dir = diff_dir or _DIFF_DIR
        self.diff_dir.mkdir(parents=True, exist_ok=True)

    def diff(self, doc_tree_v1: DocumentTree, doc_tree_v2: DocumentTree) -> DiffReport:
        """Compare deux DocumentTree et produit un DiffReport."""
        source_id = doc_tree_v1.source_id or doc_tree_v2.source_id
        v1_tag = doc_tree_v1.nodes[doc_tree_v1.root_id].version_tag or "v1"
        v2_tag = doc_tree_v2.nodes[doc_tree_v2.root_id].version_tag or "v2"

        # Représentation canonique
        canon_v1 = self._canonicalize(doc_tree_v1)
        canon_v2 = self._canonicalize(doc_tree_v2)

        changes: list[Change] = []
        all_keys = set(canon_v1.keys()) | set(canon_v2.keys())

        for key in all_keys:
            in_v1 = key in canon_v1
            in_v2 = key in canon_v2

            if in_v1 and not in_v2:
                changes.append(Change(
                    key=key,
                    change_type="removed",
                    old_content_hash=canon_v1[key].get("content_hash"),
                ))
            elif in_v2 and not in_v1:
                changes.append(Change(
                    key=key,
                    change_type="added",
                    new_content_hash=canon_v2[key].get("content_hash"),
                ))
            elif canon_v1[key].get("content_hash") != canon_v2[key].get("content_hash"):
                field_changes = self._compare_fields(canon_v1[key], canon_v2[key])
                changes.append(Change(
                    key=key,
                    change_type="modified",
                    field_changes=field_changes,
                    old_content_hash=canon_v1[key].get("content_hash"),
                    new_content_hash=canon_v2[key].get("content_hash"),
                ))

        report = DiffReport(
            source_id=source_id,
            version_from=v1_tag,
            version_to=v2_tag,
            changes=changes,
        )

        # Persistance JSON
        self._save(report)
        return report

    def load_diff(self, source_id: str, version_from: str, version_to: str) -> DiffReport | None:
        """Charge un diff précalculé depuis le disque."""
        path = self.diff_dir / f"{source_id}_{version_from}_{version_to}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DiffReport(**data)

    def _canonicalize(self, tree: DocumentTree) -> dict[str, dict[str, Any]]:
        """Extrait une représentation canonique indexée par clé sémantique."""
        result: dict[str, dict[str, Any]] = {}
        for node in tree.nodes.values():
            if node.type == NodeType.API_ENDPOINT:
                key = f"{node.metadata.get('method', 'UNKNOWN').upper()}:{node.metadata.get('path', 'unknown')}"
                result[key] = {
                    "text": node.text,
                    "markdown": node.markdown,
                    "content_hash": node.content_hash,
                    "metadata": node.metadata,
                }
            elif node.type == NodeType.API_PARAMETER:
                parent = tree.get_parent(node.id)
                parent_key = f"{parent.metadata.get('method', 'UNKNOWN').upper()}:{parent.metadata.get('path', 'unknown')}" if parent else "UNKNOWN"
                key = f"{parent_key}|param:{node.metadata.get('name', 'unknown')}"
                result[key] = {
                    "text": node.text,
                    "markdown": node.markdown,
                    "content_hash": node.content_hash,
                    "metadata": node.metadata,
                }
            elif node.type == NodeType.API_RESPONSE:
                parent = tree.get_parent(node.id)
                parent_key = f"{parent.metadata.get('method', 'UNKNOWN').upper()}:{parent.metadata.get('path', 'unknown')}" if parent else "UNKNOWN"
                key = f"{parent_key}|resp:{node.metadata.get('code', 'unknown')}"
                result[key] = {
                    "text": node.text,
                    "markdown": node.markdown,
                    "content_hash": node.content_hash,
                    "metadata": node.metadata,
                }
        return result

    def _compare_fields(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Détaille les champs modifiés entre deux entrées canoniques."""
        changes: dict[str, dict[str, Any]] = {}
        old_meta = old.get("metadata", {})
        new_meta = new.get("metadata", {})
        all_fields = set(old_meta.keys()) | set(new_meta.keys())
        for field in all_fields:
            if old_meta.get(field) != new_meta.get(field):
                changes[field] = {"old": old_meta.get(field), "new": new_meta.get(field)}
        # Si le texte/markdown a changé mais pas les métadonnées, le signaler
        # (la description d'un endpoint/paramètre vit dans markdown, jamais
        # dans metadata — sans ce check un changement de description ferait
        # varier content_hash sans qu'aucun field_changes ne l'explique)
        if old.get("text") != new.get("text") and "text" not in changes:
            changes["text"] = {"old": old.get("text"), "new": new.get("text")}
        if old.get("markdown") != new.get("markdown") and "markdown" not in changes:
            changes["markdown"] = {"old": old.get("markdown"), "new": new.get("markdown")}
        return changes

    def _save(self, report: DiffReport) -> None:
        path = self.diff_dir / f"{report.source_id}_{report.version_from}_{report.version_to}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        logger.info("diff_saved", path=str(path), changes=len(report.changes))