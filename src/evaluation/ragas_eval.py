"""Métriques qualité génération façon RAGAS — Faithfulness, Answer Relevancy,
Context Precision (sans référence), Context Recall (avec référence, sur le
sous-ensemble annoté qui en dispose).

Implémentées directement via le LLM juge local (BaseLLMClient, sortie JSON
contrainte) plutôt que la librairie `ragas` : le paquet installé (0.4.3) a
une chaîne d'import cassée dans cet environnement (`ragas.llms.base` importe
`ChatVertexAI` depuis un module qui n'existe plus dans la version installée
de langchain-community) — décision actée avec l'utilisateur d'implémenter
ces métriques nous-mêmes plutôt que de réparer la dépendance, cohérent avec
l'invariant "pas d'API payante, tout accès LLM passe par BaseLLMClient".

Modèle juge recommandé : qwen2.5:7b-instruct, forcé CPU (num_gpu=0) pour
éviter toute contention VRAM avec le pipeline évalué — voir docs/PHASE_4_SUMMARY.md §4.6.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def _judge(llm_client: BaseLLMClient, llm_config: LLMConfig, system_prompt: str, user_prompt: str) -> JudgeScore:
    config = llm_config.model_copy(update={"response_format": JudgeScore.model_json_schema()})
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    response = llm_client.complete(messages, config)
    return JudgeScore.model_validate_json(response.content)


_FAITHFULNESS_SYSTEM = (
    "Tu évalues si une réponse générée par un système RAG est fidèle au "
    "contexte documentaire fourni — c'est-à-dire que CHAQUE affirmation de "
    "la réponse peut être vérifiée dans le contexte, sans invention. "
    "Score 1.0 = entièrement fidèle, 0.0 = affirmations non présentes dans "
    "le contexte (hallucination). Réponds en JSON {score, rationale}."
)


def faithfulness(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, context_text: str, answer: str) -> JudgeScore:
    user_prompt = f"Question : {question}\n\nContexte documentaire :\n{context_text}\n\nRéponse à évaluer :\n{answer}"
    return _judge(llm_client, llm_config, _FAITHFULNESS_SYSTEM, user_prompt)


_ANSWER_RELEVANCY_SYSTEM = (
    "Tu évalues si une réponse générée traite réellement la question posée "
    "— pas si elle est correcte ou complète, seulement si elle est "
    "pertinente et sur le sujet. Score 1.0 = répond directement à la "
    "question, 0.0 = hors-sujet. Réponds en JSON {score, rationale}."
)


def answer_relevancy(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, answer: str) -> JudgeScore:
    user_prompt = f"Question : {question}\n\nRéponse : {answer}"
    return _judge(llm_client, llm_config, _ANSWER_RELEVANCY_SYSTEM, user_prompt)


_CONTEXT_RELEVANCE_SYSTEM = (
    "Tu évalues si UN passage documentaire est pertinent pour répondre à "
    "une question donnée — indépendamment de la réponse finale, juste ce "
    "passage seul. Score 1.0 = clairement pertinent, 0.0 = sans rapport. "
    "Réponds en JSON {score, rationale}."
)


def context_precision(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, chunk_texts: list[str]) -> float:
    """Context Precision sans référence (spec §9) : pertinence de chaque
    chunk jugée indépendamment, pondérée par le rang (les chunks pertinents
    en tête comptent plus — formule RAGAS standard Precision@k pondérée)."""
    if not chunk_texts:
        return 0.0
    relevances = []
    for chunk_text in chunk_texts:
        user_prompt = f"Question : {question}\n\nPassage :\n{chunk_text}"
        result = _judge(llm_client, llm_config, _CONTEXT_RELEVANCE_SYSTEM, user_prompt)
        relevances.append(1.0 if result.score >= 0.5 else 0.0)

    numerator = 0.0
    running_relevant = 0
    for i, rel in enumerate(relevances, 1):
        running_relevant += int(rel)
        precision_at_i = running_relevant / i
        numerator += precision_at_i * rel
    total_relevant = sum(relevances)
    return numerator / total_relevant if total_relevant > 0 else 0.0


_CONTEXT_RECALL_SYSTEM = (
    "Tu évalues si une réponse de référence (vérité terrain) peut être "
    "entièrement reconstituée à partir d'un contexte documentaire donné. "
    "Score 1.0 = toute l'information nécessaire est présente dans le "
    "contexte, 0.0 = rien de pertinent n'y figure. Réponds en JSON "
    "{score, rationale}."
)


def context_recall(llm_client: BaseLLMClient, llm_config: LLMConfig, expected_answer: str, context_text: str) -> JudgeScore:
    """Context Recall (spec §9) — nécessite une vérité de référence écrite,
    seulement disponible sur le sous-ensemble de questions générées par
    template (voir data/eval/questions_generated_template.jsonl)."""
    user_prompt = f"Réponse de référence :\n{expected_answer}\n\nContexte documentaire retrouvé :\n{context_text}"
    return _judge(llm_client, llm_config, _CONTEXT_RECALL_SYSTEM, user_prompt)
