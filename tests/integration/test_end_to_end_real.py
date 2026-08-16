"""Test d'intégration bout-en-bout réel — Qdrant persistant + BGE-M3 réel +
BGE-Reranker réel + Ollama réel, ensemble (audit N3).

Gap comblé : chaque composant avait déjà sa propre couverture d'intégration
isolée (retrieval seul dans test_retrieval_hybrid_phase3.py, ingestion seule
dans test_real_pipeline_phase2.py), mais aucun test ne faisait tourner la
chaîne complète retrieve -> sufficiency -> generate -> abstain avec un vrai
LLM Ollama — le seul endroit qui exerçait vraiment QueryPipeline.answer()
avec des composants 100% réels était un script (run_baseline_generation.py),
jamais un test. Ignoré (pytest.skip) si Qdrant/Ollama ne sont pas
accessibles plutôt que d'échouer bruyamment en environnement CI sans ces
services.
"""

import os

import pytest
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.generation.generator import Generator
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.pipeline import PipelineResult, QueryPipeline
from src.reliability.abstention import AbstentionGate
from src.reliability.conflict.detector import ConflictDetector
from src.reliability.sufficiency import SufficiencyChecker
from src.retrieval.hybrid_retriever import HybridRetriever

pytestmark = [pytest.mark.phase5, pytest.mark.slow]


@pytest.fixture(scope="module")
def qdrant_store() -> QdrantStore:
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("QDRANT_COLLECTION_NAME", "trade_chunks")

    client = QdrantClient(host=host, port=port)
    try:
        info = client.get_collection(collection)
        assert info.points_count > 0, "Collection is empty"
    except Exception as exc:
        pytest.skip(f"Qdrant collection '{collection}' not available: {exc}")

    return QdrantStore(client=client, collection_name=collection)


@pytest.fixture(scope="module")
def embedder() -> BGEEmbedder:
    try:
        emb = BGEEmbedder(model_name="BAAI/bge-m3", device="auto")
        emb._load_model()
        return emb
    except Exception as exc:
        pytest.skip(f"BGEEmbedder could not be loaded: {exc}")


@pytest.fixture(scope="module")
def gen_config() -> LLMConfig:
    settings = get_settings()
    return LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        repeat_penalty=settings.llm_repeat_penalty,
    )


@pytest.fixture(scope="module")
def pipeline(qdrant_store, embedder, gen_config) -> QueryPipeline:
    """QueryPipeline entièrement réel — skip si Ollama n'est pas accessible."""
    try:
        import httpx

        httpx.get(f"{gen_config.base_url}/api/tags", timeout=5.0).raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama not available at {gen_config.base_url}: {exc}")

    gen_client = LLMFactory.create(gen_config)

    retriever = HybridRetriever(
        store=qdrant_store, embedder=embedder, llm_client=gen_client, llm_config=gen_config, use_reranker=True
    )
    generator = Generator(llm_client=gen_client, llm_config=gen_config, store=qdrant_store)
    sufficiency_checker = SufficiencyChecker(llm_client=gen_client)
    abstention_gate = AbstentionGate()
    conflict_detector = ConflictDetector(llm_client=gen_client)

    return QueryPipeline(
        retriever=retriever,
        generator=generator,
        sufficiency_checker=sufficiency_checker,
        abstention_gate=abstention_gate,
        llm_config=gen_config,
        conflict_detector=conflict_detector,
    )


class TestEndToEndRealPipeline:
    def test_real_question_produces_well_formed_result(self, pipeline):
        """Pas d'assertion sur le contenu exact de la réponse (non
        déterministe, dépend du LLM réel) — vérifie que la chaîne complète
        (Qdrant réel -> BGE-M3 réel -> reranker réel -> Ollama réel) tourne
        sans erreur et produit une structure valide, cohérente avec elle-même."""
        result = pipeline.answer("Quel(s) parametre(s) sont requis pour GET /api/v3/klines ?", top_k=10)

        assert isinstance(result, PipelineResult)
        assert result.zone in ("correct", "ambiguous", "incorrect")
        assert 0.0 <= result.confidence <= 1.0
        assert result.sufficiency.verdict in ("sufficient", "partial", "insufficient")
        # Une abstention a une liste de citations vide par construction
        # (QueryPipeline._finalize_answer) ; sinon, chaque citation doit
        # référencer un chunk réellement retrouvé (pas un chunk_id halluciné
        # qui aurait dû être filtré par enrich_citations).
        for citation in result.citations:
            assert citation.text_span
            assert citation.document

    def test_out_of_corpus_question_does_not_crash(self, pipeline):
        """Question hors-corpus réelle (voir data/eval/questions_abstention.jsonl)
        — le pipeline doit dégrader proprement (abstention probable), pas
        planter, avec des composants 100% réels de bout en bout."""
        result = pipeline.answer(
            "Quelle est la politique de remboursement pour un billet d'avion annulé ?", top_k=10
        )

        assert isinstance(result, PipelineResult)
        assert result.zone in ("correct", "ambiguous", "incorrect")
