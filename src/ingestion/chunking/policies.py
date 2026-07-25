"""Chunking policies — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

"""Règles déclaratives de chunking par source — actuellement minimal.

Les politiques sont déjà exprimées dans le manifeste (ChunkingPolicy).
Ce module est réservé aux helpers de résolution de politique si besoin futur.
"""

from src.config.manifest_schema import ChunkingPolicy


def resolve_policy(manifest_policy: ChunkingPolicy) -> ChunkingPolicy:
    """Retourne la politique effective (actuellement un pass-through)."""
    return manifest_policy