"""Tests unitaires — route POST /query (spec §10, audit N1).

QueryPipeline était déjà entièrement écrit et testé (phase_4) mais n'avait
jamais de point d'appel HTTP réel — src/api/routes/query.py restait un
placeholder ("TODO: Phase 4 - implémenter"). Vérifie ici uniquement le
câblage HTTP (requête -> pipeline.answer() -> réponse), avec un
QueryPipeline factice injecté via l'override de dépendance FastAPI standard
— pas de composants réels (Qdrant/BGE-M3/Ollama), c'est un test unitaire."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_pipeline
from src.api.main import app
from src.generation.schemas import Citation
from src.pipeline import PipelineResult
from src.reliability.sufficiency import SufficiencyVerdict

pytestmark = pytest.mark.phase5


class _FakePipeline:
    def __init__(self, result: PipelineResult):
        self.result = result
        self.called_with = None

    def answer(self, question, top_k=10):
        self.called_with = {"question": question, "top_k": top_k}
        return self.result


def _sample_result(**overrides) -> PipelineResult:
    defaults = dict(
        answer="La réponse générée.",
        citations=[
            Citation(
                citation_id="cit_001",
                chunk_id=uuid4(),
                document="doc.yaml",
                text_span="span",
                support_level="fully_supported",
                claim="the claim",
            )
        ],
        confidence=0.9,
        zone="correct",
        sufficiency=SufficiencyVerdict(verdict="sufficient", confidence=0.9, method="llm_local"),
        conflicts=[],
        retrieval_metadata={"intent": "factual", "strategy": "hybrid", "latency_ms": 12.3, "candidates": 3},
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


@pytest.fixture
def client(monkeypatch):
    # build_pipeline() (appelé par le lifespan FastAPI à l'entrée du "with")
    # construit normalement Qdrant/BGE-M3/reranker/Ollama réels — inacceptable
    # dans un test unitaire. On le neutralise ; get_pipeline (utilisé par la
    # route elle-même) est overridé séparément pour pointer vers le faux
    # pipeline dont on veut vérifier les appels.
    monkeypatch.setattr("src.api.main.build_pipeline", lambda: object())

    fake = _FakePipeline(_sample_result())
    app.dependency_overrides[get_pipeline] = lambda: fake
    with TestClient(app) as test_client:
        test_client._fake_pipeline = fake  # accès direct depuis les tests
        yield test_client
    app.dependency_overrides.clear()


class TestQueryRoute:
    def test_returns_pipeline_result_as_json(self, client):
        response = client.post("/query", json={"question": "Q?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "La réponse générée."
        assert body["zone"] == "correct"
        assert len(body["citations"]) == 1

    def test_question_and_top_k_forwarded_to_pipeline(self, client):
        client.post("/query", json={"question": "Quel paramètre ?", "top_k": 5})

        assert client._fake_pipeline.called_with == {"question": "Quel paramètre ?", "top_k": 5}

    def test_top_k_defaults_to_ten(self, client):
        client.post("/query", json={"question": "Q?"})

        assert client._fake_pipeline.called_with["top_k"] == 10

    def test_missing_question_returns_422(self, client):
        response = client.post("/query", json={})

        assert response.status_code == 422

    def test_abstained_answer_has_empty_citations(self, client):
        app.dependency_overrides[get_pipeline] = lambda: _FakePipeline(
            _sample_result(
                answer="Information non trouvée dans la documentation.",
                citations=[],
                zone="incorrect",
                sufficiency=SufficiencyVerdict(verdict="insufficient", confidence=0.1, method="llm_local"),
            )
        )
        response = client.post("/query", json={"question": "Q?"})

        assert response.status_code == 200
        assert response.json()["citations"] == []
        assert response.json()["zone"] == "incorrect"
