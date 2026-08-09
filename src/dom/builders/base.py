"""Interface abstraite pour tous les builders DOM."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.config.manifest_schema import SourceManifest
from src.dom.models import DocumentTree


class AbstractDOMBuilder(ABC):
    """Builder produisant un DocumentTree unifié depuis un fichier source."""

    @abstractmethod
    def build(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        """Parse le fichier source et retourne un DocumentTree complet."""
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Retourne True si ce builder peut traiter le fichier."""
        ...

    def _create_root_node(self, source_path: str, manifest: SourceManifest):
        """Crée le nœud racine, en y attachant version_tag/version_order et
        validity (valid_from/valid_until/status) déclarés dans le manifeste.

        Portés uniquement par la racine, pas par chaque nœud individuel : un
        fichier source représente une seule version / une seule validité
        (déclarée une fois dans le manifeste), jamais différente d'un
        paragraphe/endpoint à l'autre du même fichier — voir
        HierarchicalChunker qui hérite ces valeurs depuis la racine de
        l'arbre au moment du chunking.

        Corrige un gap qui existait depuis la Phase 2 : aucun builder ne
        propageait manifest.versioning/manifest.validity, donc tous les
        chunks ingérés avaient version_tag=None et status=None quel que soit
        ce que déclarait le manifeste (voir docs/PHASE_3_SUMMARY.md §6 et
        docs/PHASE_4_SUMMARY.md — Tâche 1)."""
        from src.dom.models import DOMNode, NodeType

        version_tag, version_order = self._resolve_version(source_path, manifest)

        return DOMNode(
            type=NodeType.DOCUMENT,
            source_id=manifest.source_id,
            source_path=source_path,
            is_structural=True,
            is_content=False,
            version_tag=version_tag,
            version_order=version_order,
            valid_from=manifest.validity.valid_from,
            valid_until=manifest.validity.valid_until,
            status=manifest.validity.status,
        )

    @staticmethod
    def _resolve_version(source_path: str, manifest: SourceManifest) -> tuple[str | None, int | None]:
        """Dérive (version_tag, version_order) depuis manifest.versioning
        appliqué au nom de fichier. Seule stratégie supportée actuellement :
        filename_pattern (spec §13) — les autres stratégies déclarées
        (directory_pattern, header_pattern) retournent (None, None), pas
        d'erreur, pour ne jamais faire échouer l'ingestion d'une source qui
        ne se versionne pas ainsi."""
        import re

        versioning = manifest.versioning
        if versioning.strategy != "filename_pattern" or not versioning.pattern:
            return None, None

        match = re.search(versioning.pattern, Path(source_path).name)
        if not match:
            return None, None

        tag = match.group("version")
        order = (
            versioning.order.index(tag)
            if versioning.order and tag in versioning.order
            else None
        )
        return tag, order
