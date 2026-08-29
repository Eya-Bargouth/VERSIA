"""Schémas Pydantic requête API (spec §10).

La réponse réutilise directement `PipelineResult` (src/pipeline.py) comme
`response_model` — pas de duplication d'un schéma de réponse séparé pour
une structure qui existe déjà et est déjà celle produite par
`QueryPipeline.answer()`.
"""

from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


class IngestRequest(BaseModel):
    """Requête POST /ingest.

    La spec §10 documente un champ `manifest_dir` — obsolète depuis la
    suppression des manifestes (CLAUDE.md, refactor Phase 5 : sources
    auto-découvertes depuis `raw_dir`). `versioning_dir` est son équivalent
    réel : la seule exception déclarable restante (pattern de version d'une
    source versionnée), jamais un manifeste par source.
    """

    raw_dir: str
    versioning_dir: str | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    estimated_duration: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    report: dict | None = None
    error: str | None = None
