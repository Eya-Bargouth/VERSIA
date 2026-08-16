"""Application FastAPI (spec §10)."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api.dependencies import build_pipeline
from src.api.routes.query import router as query_router
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("api_startup", version="2.1.0", llm_provider=settings.llm_provider)
    # Construit une seule fois tous les composants coûteux (BGE-M3, reranker,
    # connexions Qdrant/Ollama) — jamais reconstruits à chaque requête.
    app.state.pipeline = build_pipeline()
    yield
    logger.info("api_shutdown")


app = FastAPI(title="TRADE API", version="2.1.0", lifespan=lifespan)
app.include_router(query_router)


@app.get("/health")
async def health():
    settings = get_settings()
    services = {
        "qdrant": "unknown",
        "llm": "unknown",
    }
    return {
        "status": "healthy",
        "services": services,
        "version": "2.1.0",
    }
