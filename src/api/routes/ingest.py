"""Route POST /ingest (spec §10) — lance l'ingestion en tâche de fond,
réutilise store/embedder déjà chargés dans app.state.pipeline (pas de
rechargement de BGE-M3) et run_ingestion() de src/ingestion/pipeline.py,
la même fonction que scripts/run_ingestion.py.

Store de jobs en mémoire (dict sur app.state) : pas de broker de messages
(spec §3), donc pas de persistance non plus — l'état d'un job est perdu au
redémarrage de l'API. Acceptable pour ce périmètre ; l'ingestion peut de
toute façon être relancée par script.
"""

import re
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile

from src.api.cache import QueryCache
from src.api.dependencies import get_cache, get_pipeline
from src.api.schemas import IngestJobStatus, IngestRequest, IngestResponse
from src.ingestion.pipeline import run_ingestion
from src.pipeline import QueryPipeline

logger = structlog.get_logger(__name__)
router = APIRouter()

# N'autorise que des noms de dossier sûrs (pas de "..", pas de séparateur de
# chemin) — source_id vient d'un champ texte libre côté UI (voir POST
# /ingest/upload) et sert directement à construire un chemin sous raw/.
_SOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _run_job(
    jobs: dict,
    job_id: str,
    pipeline: QueryPipeline,
    cache: QueryCache,
    raw_dir: Path,
    versioning_dir: Path | None,
    source_id_filter: str | None = None,
) -> None:
    jobs[job_id]["status"] = "running"
    try:
        report = run_ingestion(
            raw_dir=raw_dir,
            store=pipeline.retriever.store,
            embedder=pipeline.retriever.embedder,
            versioning_dir=versioning_dir,
            source_id_filter=source_id_filter,
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


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    source_id: str = Form(...),
    pipeline: QueryPipeline = Depends(get_pipeline),
    cache: QueryCache = Depends(get_cache),
) -> IngestResponse:
    """Ajoute un nouveau document au corpus depuis l'UI (§ Ingestion, bouton
    upload) : écrit le fichier sous raw/<source_id>/, puis lance la même
    ingestion que POST /ingest, restreinte à cette seule source
    (source_id_filter) pour ne pas retraiter tout le corpus existant.

    Seul écrivain de raw/ dans tout le projet — action utilisateur explicite
    via ce endpoint, pas une écriture silencieuse du pipeline d'ingestion
    lui-même (voir CLAUDE.md, invariant "raw/ lecture seule").
    """
    source_id = source_id.strip()
    if not _SOURCE_ID_RE.match(source_id):
        raise HTTPException(
            status_code=400,
            detail="source_id ne doit contenir que des lettres, chiffres, '-' ou '_' (pas de chemin).",
        )
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")

    raw_root = Path("raw")
    dest_dir = raw_root / source_id
    dest_path = dest_dir / filename
    if dest_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"{dest_path} existe déjà — renommez le fichier ou supprimez l'existant avant de réuploader.",
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest_path.write_bytes(content)
    logger.info("ingest_upload_saved", source_id=source_id, filename=filename, size_bytes=len(content))

    job_id = str(uuid.uuid4())
    request.app.state.jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(
        _run_job,
        request.app.state.jobs,
        job_id,
        pipeline,
        cache,
        raw_root,
        None,
        source_id,
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
