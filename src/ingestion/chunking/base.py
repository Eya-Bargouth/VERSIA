"""Chunking base — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

"""Interface abstraite pour les stratégies de chunking."""

from abc import ABC, abstractmethod

from src.config.manifest_schema import ChunkingPolicy
from src.dom.models import DocumentTree
from src.ingestion.metadata import Chunk


class ChunkingStrategy(ABC):
    """Stratégie de découpage d'un DocumentTree en chunks."""

    @abstractmethod
    def chunk(self, tree: DocumentTree, policy: ChunkingPolicy) -> list[Chunk]:
        """Découpe l'arbre en chunks selon la politique déclarative."""
        ...