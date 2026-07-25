"""Application FastAPI minimale — Phase 1."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("api_startup", version="2.1.0", llm_provider=settings.llm_provider)
    yield
    logger.info("api_shutdown")


app = FastAPI(title="TRADE API", version="2.1.0", lifespan=lifespan)


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
