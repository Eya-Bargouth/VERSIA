"""Application FastAPI (spec §10)."""

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import build_cache, build_pipeline
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config.settings import get_settings

_STATIC_DIR = Path(__file__).parent / "static"

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("api_startup", version="2.1.0", llm_provider=settings.llm_provider)
    # Construit une seule fois tous les composants coûteux (BGE-M3, reranker,
    # connexions Qdrant/Ollama) — jamais reconstruits à chaque requête.
    app.state.pipeline = build_pipeline()
    # Store de jobs /ingest en mémoire — voir src/api/routes/ingest.py.
    app.state.jobs = {}
    # Cache Redis des réponses /query (hors périmètre spec v2.1, voir
    # src/api/cache.py) — dégrade silencieusement si Redis est injoignable.
    app.state.cache = build_cache()
    yield
    logger.info("api_shutdown")


app = FastAPI(title="TRADE API", version="2.1.0", lifespan=lifespan)
app.include_router(query_router)
app.include_router(health_router)
app.include_router(ingest_router)
# Interface de vérification manuelle des fonctionnalités (hors périmètre de
# la spec v2.1, ajout demandé séparément). Montée sous /ui, pas /, pour ne
# jamais entrer en conflit avec les routes API existantes (/query, /health,
# /ingest).
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
