"""Schémas Pydantic requête API (spec §10).

La réponse réutilise directement `PipelineResult` (src/pipeline.py) comme
`response_model` — pas de duplication d'un schéma de réponse séparé pour
une structure qui existe déjà et est déjà celle produite par
`QueryPipeline.answer()`.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
