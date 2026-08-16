"""Dépendances injectables FastAPI (spec §10).

`build_pipeline()` construit une seule fois, au démarrage de l'API (voir
`lifespan` dans main.py), tous les composants coûteux à charger (BGE-M3,
reranker, connexions Qdrant/Ollama) — jamais reconstruits à chaque requête.
Reprend exactement le même agencement de composants que
`scripts/run_baseline_generation.py::build_components()`, seul déploiement
réel de QueryPipeline avant celui-ci.
"""

from pathlib import Path

from fastapi import Request
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.generation.generator import Generator
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.llm.providers import vllm_client  # noqa: F401 — self-registers
from src.pipeline import QueryPipeline
from src.reliability.abstention import AbstentionGate
from src.reliability.conflict.detector import ConflictDetector
from src.reliability.sufficiency import SufficiencyChecker
from src.retrieval.hybrid_retriever import HybridRetriever

_PROJECT_ROOT = Path(__file__).parents[2]


def build_pipeline() -> QueryPipeline:
    settings = get_settings()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        dtype=settings.embedding_dtype,
        backend=settings.embedding_backend,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    gen_config = LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        repeat_penalty=settings.llm_repeat_penalty,
    )
    gen_client = LLMFactory.create(gen_config)

    retriever = HybridRetriever(
        store=store, embedder=embedder, llm_client=gen_client, llm_config=gen_config, use_reranker=True
    )
    retriever.warm_up()  # coût de chargement du reranker payé ici, pas sur la première requête HTTP

    generator = Generator(llm_client=gen_client, llm_config=gen_config, store=store)
    sufficiency_checker = SufficiencyChecker(llm_client=gen_client)
    abstention_gate = AbstentionGate()
    conflict_detector = ConflictDetector(llm_client=gen_client, diff_dir=_PROJECT_ROOT / "data" / "diffs")

    return QueryPipeline(
        retriever=retriever,
        generator=generator,
        sufficiency_checker=sufficiency_checker,
        abstention_gate=abstention_gate,
        llm_config=gen_config,
        conflict_detector=conflict_detector,
    )


def get_pipeline(request: Request) -> QueryPipeline:
    """Dépendance FastAPI — lit le pipeline construit au démarrage
    (app.state.pipeline, voir lifespan dans main.py), jamais reconstruit
    par requête."""
    return request.app.state.pipeline
