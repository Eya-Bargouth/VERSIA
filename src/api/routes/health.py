"""Route GET /health (spec §10) — réutilise Settings.check_services(), la
même logique de ping que validate_startup() appelée au démarrage de l'API
(voir src/api/main.py::lifespan), pour ne pas dupliquer les URLs par
provider."""

from fastapi import APIRouter

from src.config.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    services = settings.check_services()
    status = "healthy" if all(v == "ok" for v in services.values()) else "degraded"
    return {"status": status, "services": services, "version": "2.1.0"}
