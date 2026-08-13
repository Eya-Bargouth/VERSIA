"""Interface abstraite pour tous les builders DOM."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.config.source_config import SourceConfig
from src.dom.models import DocumentTree


class AbstractDOMBuilder(ABC):
    """Builder produisant un DocumentTree unifié depuis un fichier source."""

    @abstractmethod
    def build(self, source_path: str, config: SourceConfig) -> DocumentTree:
        """Parse le fichier source et retourne un DocumentTree complet."""
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Retourne True si ce builder peut traiter le fichier."""
        ...

    def _create_root_node(self, source_path: str, config: SourceConfig):
        """Crée le nœud racine, en y attachant version_tag/version_order
        déduits de config.version_pattern (si déclaré pour cette source).

        Portés uniquement par la racine, pas par chaque nœud individuel : un
        fichier source représente une seule version, jamais différente d'un
        paragraphe/endpoint à l'autre du même fichier — voir
        HierarchicalChunker qui hérite ces valeurs depuis la racine de
        l'arbre au moment du chunking.

        La validité temporelle (valid_from/valid_until/status) n'est plus
        déclarable par source (ancien ValidityConfig, supprimé) — toute
        source est "active" par défaut, sans date d'expiration. Seul le
        versioning reste une exception déclarable (voir SourceConfig)."""
        from src.dom.models import DOMNode, NodeType

        version_tag, version_order = self._resolve_version(source_path, config)

        return DOMNode(
            type=NodeType.DOCUMENT,
            source_id=config.source_id,
            source_path=source_path,
            is_structural=True,
            is_content=False,
            version_tag=version_tag,
            version_order=version_order,
            status="active",
        )

    @staticmethod
    def _resolve_version(source_path: str, config: SourceConfig) -> tuple[str | None, int | None]:
        """Dérive (version_tag, version_order) depuis config.version_pattern
        appliqué au nom de fichier. Retourne (None, None) si aucun pattern
        n'est déclaré pour cette source (cas normal, non versionné) — jamais
        d'erreur, une source ne se versionnant pas ainsi ingère quand même."""
        import re

        if not config.version_pattern:
            return None, None

        match = re.search(config.version_pattern, Path(source_path).name)
        if not match:
            return None, None

        tag = match.group("version")
        order = (
            config.version_order.index(tag)
            if config.version_order and tag in config.version_order
            else None
        )
        return tag, order
