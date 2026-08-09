"""Metadata et Chunk — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

"""Schémas Pydantic pour les chunks et leurs métadonnées."""

from datetime import date
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.dom.models import NodeType


class ChunkMetadata(BaseModel):
    """Métadonnées typées d'un chunk."""

    source_type: str          # libre, jamais un enum figé
    source_id: str
    node_type: NodeType
    hierarchy_path: str
    format_original: str
    # Chemin du fichier source réel (ex. "raw/specs-api/stripe/spec3-v2323.yaml"),
    # distinct de source_id (l'identifiant de corpus, ex. "stripe_specs") — un
    # même source_id peut couvrir plusieurs fichiers (versions, cheat sheets...).
    # Défaut "" pour rester compatible avec les chunks déjà en cache/Qdrant
    # ingérés avant l'ajout de ce champ (ré-ingestion nécessaire pour le peupler).
    source_path: str = ""
    section_title: str | None = None
    page_num: int | None = None
    line_num: int | None = None
    char_offset: int | None = None


class Chunk(BaseModel):
    """Unité d'indexation — chunk hiérarchique avec Contextual Retrieval."""

    chunk_id: UUID = Field(default_factory=uuid4)
    source_id: str
    node_ids: list[str] = Field(default_factory=list)
    text: str                        # contextual_prefix + raw_text
    raw_text: str                    # contenu sans préfixe
    contextual_prefix: str
    parent_chunk_id: UUID | None = None
    child_chunk_ids: list[UUID] = Field(default_factory=list)
    level: int = 0
    metadata: ChunkMetadata
    hierarchy_path: str
    version_tag: str | None = None
    version_order: int | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    status: str | None = None
    content_hash: str | None = None  # SHA-256 du texte final, pour idempotence