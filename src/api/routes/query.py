"""Route POST /query (spec §10) — point d'entrée HTTP unique vers
QueryPipeline, la même orchestration retrieve->sufficiency->generate->
conflict->abstain déjà exercée par scripts/run_baseline_generation.py.
"""

from fastapi import APIRouter, Depends

from src.api.cache import QueryCache
from src.api.dependencies import get_cache, get_pipeline
from src.api.schemas import QueryRequest
from src.pipeline import PipelineResult, QueryPipeline

router = APIRouter()


@router.post("/query", response_model=PipelineResult)
def query(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_pipeline),
    cache: QueryCache = Depends(get_cache),
) -> PipelineResult:
    # Handler synchrone (pas "async def") délibérément : QueryPipeline.answer()
    # est une chaîne d'appels bloquants (httpx synchrone vers Qdrant/Ollama,
    # parfois plusieurs dizaines de secondes). FastAPI exécute automatiquement
    # les handlers "def" dans un threadpool — en "async def", cet appel
    # bloquant gèlerait la boucle d'événements pour toutes les requêtes
    # concurrentes.
    cached = cache.get(request.question, request.top_k)
    if cached is not None:
        return cached

    result = pipeline.answer(request.question, top_k=request.top_k)
    cache.set(request.question, request.top_k, result)
    return result
