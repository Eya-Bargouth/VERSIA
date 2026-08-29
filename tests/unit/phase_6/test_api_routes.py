"""Tests unitaires des routes API Phase 6 — pipeline, run_ingestion() et
Settings.check_services() mockés (BGE-M3, reranker, Qdrant, Ollama : les
dépendances lourdes de CLAUDE.md), donc tests/unit/ et pas
tests/integration/. Vérifient uniquement le câblage HTTP (schémas, cycle de
vie des jobs /ingest, codes de statut) — le pipeline réel de bout en bout
est déjà couvert par tests/integration/test_end_to_end_real.py et
test_real_pipeline_phase2.py.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
import src.api.routes.ingest as ingest_route
from src.ingestion.pipeline import IngestionReport

pytestmark = pytest.mark.phase6


class _FakePipeline:
    def __init__(self):
        self.retriever = SimpleNamespace(store=object(), embedder=object())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "build_pipeline", lambda: _FakePipeline())
    with TestClient(api_main.app) as c:
        yield c


class TestHealth:
    def test_healthy_when_all_services_ok(self, client, monkeypatch):
        monkeypatch.setattr(
            "src.api.routes.health.get_settings",
            lambda: SimpleNamespace(check_services=lambda: {"qdrant": "ok", "llm": "ok"}),
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "healthy",
            "services": {"qdrant": "ok", "llm": "ok"},
            "version": "2.1.0",
        }

    def test_degraded_when_one_service_down(self, client, monkeypatch):
        monkeypatch.setattr(
            "src.api.routes.health.get_settings",
            lambda: SimpleNamespace(check_services=lambda: {"qdrant": "error", "llm": "ok"}),
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestIngest:
    def test_ingest_returns_job_id_and_transitions_to_done(self, client, monkeypatch, tmp_path):
        fake_report = IngestionReport(total_documents=1, total_chunks=3)
        monkeypatch.setattr(ingest_route, "run_ingestion", lambda **kwargs: fake_report)

        resp = client.post("/ingest", json={"raw_dir": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        job_id = body["job_id"]

        # Starlette exécute les BackgroundTasks avant de renvoyer la réponse
        # au client ASGI — le job est donc déjà terminé au retour de .post().
        status_resp = client.get(f"/ingest/{job_id}")
        assert status_resp.status_code == 200
        status_body = status_resp.json()
        assert status_body["status"] == "done"
        assert status_body["report"]["total_chunks"] == 3

    def test_ingest_job_failure_is_reported(self, client, monkeypatch, tmp_path):
        def _boom(**kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(ingest_route, "run_ingestion", _boom)

        resp = client.post("/ingest", json={"raw_dir": str(tmp_path)})
        job_id = resp.json()["job_id"]

        status_body = client.get(f"/ingest/{job_id}").json()
        assert status_body["status"] == "failed"
        assert "disk on fire" in status_body["error"]

    def test_unknown_job_id_is_404(self, client):
        resp = client.get("/ingest/does-not-exist")
        assert resp.status_code == 404
