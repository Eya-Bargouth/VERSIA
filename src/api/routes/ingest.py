"""Route POST /ingest (spec §10) — lance l'ingestion en tâche de fond,
réutilise store/embedder déjà chargés dans app.state.pipeline (pas de
rechargement de BGE-M3) et run_ingestion() de src/ingestion/pipeline.py,
la même fonction que scripts/run_ingestion.py.

Store de jobs en mémoire (dict sur app.state) : pas de broker de messages
(spec §3), donc pas de persistance non plus — l'état d'un job est perdu au
redémarrage de l'API. Acceptable pour ce périmètre ; l'ingestion peut de
toute façon être relancée par script.
"""

import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from src.api.cache import QueryCache
from src.api.dependencies import get_cache, get_pipeline
from src.api.schemas import IngestJobStatus, IngestRequest, IngestResponse
from src.ingestion.pipeline import run_ingestion
from src.pipeline import QueryPipeline

logger = structlog.get_logger(__name__)
router = APIRouter()


def _run_job(
    jobs: dict, job_id: str, pipeline: QueryPipeline, cache: QueryCache, raw_dir: Path, versioning_dir: Path | None
) -> None:
    jobs[job_id]["status"] = "running"
    try:
        report = run_ingestion(
            raw_dir=raw_dir,
            store=pipeline.retriever.store,
            embedder=pipeline.retriever.embedder,
            versioning_dir=versioning_dir,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["report"] = report.model_dump()
        # Les réponses en cache peuvent référencer des chunks désormais
        # modifiés/supprimés par cette ré-ingestion.
        cache.clear()
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        logger.error("ingest_job_failed", job_id=job_id, error=str(exc))


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    pipeline: QueryPipeline = Depends(get_pipeline),
    cache: QueryCache = Depends(get_cache),
) -> IngestResponse:
    job_id = str(uuid.uuid4())
    request.app.state.jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(
        _run_job,
        request.app.state.jobs,
        job_id,
        pipeline,
        cache,
        Path(payload.raw_dir),
        Path(payload.versioning_dir) if payload.versioning_dir else None,
    )
    return IngestResponse(
        job_id=job_id,
        status="queued",
        estimated_duration="unknown — dépend de la taille du corpus, suivre via GET /ingest/{job_id}",
    )


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
def ingest_status(job_id: str, request: Request) -> IngestJobStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No ingestion job with id {job_id}")
    return IngestJobStatus(job_id=job_id, **job)
