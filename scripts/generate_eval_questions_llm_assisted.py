#!/usr/bin/env python3
"""Phase 5 — génère des questions candidates "multi-sources" et "ambiguës"
en s'appuyant sur le vrai retrieval (pas des suppositions sur le corpus) :
pour chaque thème candidat, on interroge le vrai HybridRetriever (BGE-M3 +
Qdrant réel) et on ne garde le thème que si des chunks de >=2 source_id
distincts apparaissent réellement dans les résultats. Le LLM local
(qwen2.5:7b-instruct) formule ensuite UNE question naturelle à partir des
passages réellement retrouvés.

Sortie : data/eval/questions_candidates_llm_review.jsonl — un CANDIDAT par
thème confirmé, avec les passages sources inclus pour audit. Ne PAS fusionner
directement dans questions_v1.jsonl : la spec (§9) exige un audit qualitatif
manuel du jeu de test avant toute évaluation RAGAS automatique.
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig, LLMMessage
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.retrieval.hybrid_retriever import HybridRetriever

OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_candidates_llm_review.jsonl"

# Thèmes candidats — chacun étiqueté avec l'intention de catégorie visée,
# mais la confirmation "vraiment multi-source" vient du VRAI retrieval
# ci-dessous, pas de cette liste seule.
SEED_THEMES = [
    ("authentification à l'API", "multi_source"),
    ("vérification de signature de webhook", "multi_source"),
    ("validation des entrées utilisateur", "multi_source"),
    ("protection contre les attaques par injection", "multi_source"),
    ("gestion des erreurs et codes de réponse", "multi_source"),
    ("protection des données personnelles", "multi_source"),
    ("limite de rate limiting de l'API", "ambiguous"),
    ("politique de pagination des résultats", "ambiguous"),
    ("durée de conservation des données", "ambiguous"),
    ("format de réponse en cas d'erreur", "ambiguous"),
    ("expiration et rotation des clés d'API", "ambiguous"),
    ("gestion des incidents et interruptions de service", "multi_source"),
]

MIN_DISTINCT_SOURCES = 2
TOP_K = 8


class DraftedQuestion(BaseModel):
    question: str
    rationale: str
    sources_needed: list[str]


DRAFT_SYSTEM_PROMPT = (
    "Tu formules UNE question de test pour évaluer un système de question-réponse "
    "documentaire. On te donne un thème et des extraits réels retrouvés dans "
    "plusieurs sources documentaires différentes. Rédige une question naturelle, "
    "en français, qu'un utilisateur poserait réellement, et qui nécessite "
    "vraiment de croiser ou comparer les sources fournies pour y répondre "
    "correctement (pas une question à laquelle une seule source suffit). "
    "Réponds uniquement avec un JSON valide correspondant au schéma demandé."
)


def build_retriever() -> HybridRetriever:
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
    return HybridRetriever(store=store, embedder=embedder, use_reranker=True)


def confirm_multi_source(retriever: HybridRetriever, theme: str) -> list[dict] | None:
    result = retriever.retrieve(query=theme, top_k=TOP_K, k_dense=20, k_sparse=20)
    chunks = result.get("results", [])
    by_source: dict[str, dict] = {}
    for c in chunks:
        sid = c.get("payload", {}).get("source_id", "unknown")
        if sid not in by_source:
            by_source[sid] = c
    if len(by_source) < MIN_DISTINCT_SOURCES:
        return None
    return list(by_source.values())


def draft_question(llm_client, llm_config: LLMConfig, theme: str, evidence: list[dict]) -> DraftedQuestion | None:
    passages = "\n\n".join(
        f"[Source: {c['payload'].get('source_id')} | {c['payload'].get('source_path')}]\n"
        f"{c['payload'].get('text', '')[:600]}"
        for c in evidence
    )
    user_message = f"Thème : {theme}\n\nExtraits retrouvés :\n\n{passages}"
    messages = [
        LLMMessage(role="system", content=DRAFT_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_message),
    ]
    config = llm_config.model_copy(update={"response_format": DraftedQuestion.model_json_schema()})
    response = llm_client.complete(messages, config)
    try:
        return DraftedQuestion.model_validate_json(response.content)
    except Exception:
        return None


def main():
    settings = get_settings()
    retriever = build_retriever()

    llm_config = LLMConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct",
        base_url=settings.llm_base_url,
        temperature=0.2,
        max_tokens=600,
        num_gpu=0,  # juge/rédacteur hors ligne — évite toute contention VRAM
    )
    llm_client = LLMFactory.create(llm_config)

    candidates = []
    for theme, intent in SEED_THEMES:
        evidence = confirm_multi_source(retriever, theme)
        if evidence is None:
            print(f"[skip] '{theme}' — moins de {MIN_DISTINCT_SOURCES} sources distinctes retrouvées")
            continue
        drafted = draft_question(llm_client, llm_config, theme, evidence)
        if drafted is None:
            print(f"[skip] '{theme}' — échec de parsing de la sortie LLM")
            continue
        candidates.append(
            {
                "question": drafted.question,
                "expected_answer": None,  # pas de vérité de référence écrite (RAGAS sans référence)
                "expected_sources": [c["payload"].get("source_path") for c in evidence],
                "category": intent,
                "eval_criteria": "synthese_multi_sources" if intent == "multi_source" else "coherence_inter_sources",
                "requires_obsolescence_check": False,
                "_generation": {
                    "method": "llm_assisted",
                    "seed_theme": theme,
                    "rationale": drafted.rationale,
                    "sources_needed": drafted.sources_needed,
                    "evidence_preview": [
                        {
                            "source_id": c["payload"].get("source_id"),
                            "source_path": c["payload"].get("source_path"),
                            "text_preview": c["payload"].get("text", "")[:300],
                        }
                        for c in evidence
                    ],
                },
            }
        )
        print(f"[ok] '{theme}' -> {drafted.question}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{len(candidates)} candidats écrits dans {OUT_PATH} — AUDIT MANUEL REQUIS avant fusion.")


if __name__ == "__main__":
    main()
