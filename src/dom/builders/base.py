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

    def _create_root_node(self, source_path: str, source_id: str):
        from src.dom.models import DOMNode, NodeType

        return DOMNode(
            type=NodeType.DOCUMENT,
            source_id=source_id,
            source_path=source_path,
            is_structural=True,
            is_content=False,
        )
