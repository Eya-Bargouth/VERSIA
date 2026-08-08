"""Types partagés par les détecteurs de conflits (spec §14.9)."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ConflictReport(BaseModel):
    """Verdict unifié d'un détecteur de conflit, quelle que soit sa méthode."""

    conflict: bool
    type: Literal["factual", "temporal", "opinion"] | None = None
    confidence: float | None = None
    method: Literal["nli", "llm_fallback", "version_diff"]
    chunks: list[UUID] = Field(default_factory=list)
    explanation: str | None = None
