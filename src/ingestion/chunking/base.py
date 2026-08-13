"""Interface abstraite pour les stratégies de chunking."""

from abc import ABC, abstractmethod

from src.dom.models import DocumentTree
from src.ingestion.metadata import Chunk


class ChunkingStrategy(ABC):
    """Stratégie de découpage d'un DocumentTree en chunks."""

    @abstractmethod
    def chunk(self, tree: DocumentTree, source_type: str = "unknown") -> list[Chunk]:
        """Découpe l'arbre en chunks — un chunk par nœud de contenu
        (is_content=True), universellement, quel que soit le format ou le
        domaine documentaire (plus de politique déclarative par source)."""
        ...
