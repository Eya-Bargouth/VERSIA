"""Route POST /query (spec §10) — point d'entrée HTTP unique vers
QueryPipeline, la même orchestration retrieve->sufficiency->generate->
conflict->abstain déjà exercée par scripts/run_baseline_generation.py.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.schemas import QueryRequest
from src.pipeline import PipelineResult, QueryPipeline

router = APIRouter()


@router.post("/query", response_model=PipelineResult)
def query(request: QueryRequest, pipeline: QueryPipeline = Depends(get_pipeline)) -> PipelineResult:
    # Handler synchrone (pas "async def") délibérément : QueryPipeline.answer()
    # est une chaîne d'appels bloquants (httpx synchrone vers Qdrant/Ollama,
    # parfois plusieurs dizaines de secondes). FastAPI exécute automatiquement
    # les handlers "def" dans un threadpool — en "async def", cet appel
    # bloquant gèlerait la boucle d'événements pour toutes les requêtes
    # concurrentes.
    return pipeline.answer(request.question, top_k=request.top_k)
