"""Stratégies de chunking."""
from src.ingestion.chunking.base import ChunkingStrategy
from src.ingestion.chunking.hierarchical import HierarchicalChunker

__all__ = ["ChunkingStrategy", "HierarchicalChunker"]