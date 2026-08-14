"""Métriques qualité génération façon RAGAS — Faithfulness, Answer Relevancy,
Context Precision, Context Recall.

Implémentées directement via le LLM juge local (BaseLLMClient, sortie JSON
contrainte) plutôt que la librairie `ragas` : le paquet (0.4.3, dernière
version dispo) a une chaîne d'import cassée dans cet environnement
(`ragas.llms.base` importe `ChatVertexAI` depuis un module absent de
langchain-community, reproduit à l'identique lors d'une réinstallation à
neuf) — décision actée avec l'utilisateur d'implémenter ces métriques
nous-mêmes plutôt que de réparer la dépendance, cohérent avec l'invariant
"pas d'API payante, tout accès LLM passe par BaseLLMClient".

Les algorithmes ci-dessous répliquent fidèlement la méthode réelle de ragas
(lue directement dans `ragas/metrics/_faithfulness.py`,
`_answer_relevance.py`, `_context_precision.py`, `_context_recall.py` du
paquet installé) plutôt qu'un jugement LLM holistique portant le même nom :

- Faithfulness : décomposition de la réponse en affirmations atomiques
  (claims), puis vérification NLI de chaque claim contre le contexte —
  score = proportion de claims vérifiées. PAS un seul jugement global.
- Answer Relevancy : génération de questions synthétiques à partir de la
  réponse, puis similarité cosinus (embeddings, pas jugement LLM) entre la
  question originale et les questions générées — pénalisé si la réponse est
  évasive ("noncommittal"). Réutilise l'embedder BGE-M3 déjà chargé pour
  l'indexation, pas de nouvelle dépendance d'embedding.
- Context Precision : Average Precision pondérée par le rang, sur des
  verdicts LLM d'utilité de chaque chunk *vis-à-vis d'une réponse donnée*
  (`expected_answer` si disponible, sinon la réponse générée) — pas une
  pertinence de passage jugée dans l'absolu.
- Context Recall : classification phrase par phrase de la réponse de
  référence, attribuée ou non au contexte retrouvé — pas un score global.

Simplification assumée par rapport à ragas : un seul appel LLM par étape
(pas d'auto-ensembling multi-échantillons ni de génération multi-appels des
questions synthétiques) — le juge tourne en local sur CPU (qwen2.5:7b),
répliquer l'ensembling de ragas multiplierait le coût sans changer la
formule de score. Documenté ici plutôt que fait silencieusement.

Modèle juge recommandé : qwen2.5:7b-instruct, forcé CPU (num_gpu=0) pour
éviter toute contention VRAM avec le pipeline évalué — voir docs/PHASE_4_SUMMARY.md §4.6.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def _call(llm_client: BaseLLMClient, llm_config: LLMConfig, system_prompt: str, user_prompt: str, output_model: type[BaseModel]):
    config = llm_config.model_copy(update={"response_format": output_model.model_json_schema()})
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    response = llm_client.complete(messages, config)
    return output_model.model_validate_json(response.content)


# ---------------------------------------------------------------------------
# Faithfulness — décomposition en claims + vérification NLI par claim
# (ragas: StatementGeneratorPrompt puis NLIStatementPrompt)
# ---------------------------------------------------------------------------

class _Statements(BaseModel):
    statements: list[str]


class _StatementVerdict(BaseModel):
    statement: str
    reason: str
    verdict: int = Field(ge=0, le=1)


class _NLIVerdicts(BaseModel):
    statements: list[_StatementVerdict]


_STATEMENT_GENERATOR_SYSTEM = (
    "Décompose la réponse suivante en affirmations atomiques et "
    "compréhensibles individuellement (une par fait vérifiable). N'utilise "
    "aucun pronom — remplace-le par ce qu'il désigne. Réponds en JSON "
    "{statements: [...]}."
)

_NLI_SYSTEM = (
    "Pour chaque affirmation fournie, détermine si elle peut être déduite "
    "directement du contexte documentaire donné. verdict=1 si oui, verdict=0 "
    "si l'affirmation n'est pas présente ou contredite par le contexte. "
    "Réponds en JSON {statements: [{statement, reason, verdict}, ...]}, une "
    "entrée par affirmation reçue, dans le même ordre."
)


def faithfulness(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, context_text: str, answer: str) -> JudgeScore:
    gen_prompt = f"Question : {question}\n\nRéponse à décomposer :\n{answer}"
    statements = _call(llm_client, llm_config, _STATEMENT_GENERATOR_SYSTEM, gen_prompt, _Statements).statements
    if not statements:
        return JudgeScore(score=0.0, rationale="Aucune affirmation vérifiable extraite de la réponse.")

    nli_prompt = (
        f"Contexte documentaire :\n{context_text}\n\n"
        f"Affirmations à juger (dans l'ordre) :\n" + "\n".join(f"- {s}" for s in statements)
    )
    verdicts = _call(llm_client, llm_config, _NLI_SYSTEM, nli_prompt, _NLIVerdicts).statements

    faithful = sum(1 for v in verdicts if v.verdict)
    total = len(verdicts) or 1
    unsupported = [v.statement for v in verdicts if not v.verdict]
    rationale = (
        f"{faithful}/{total} affirmations vérifiées dans le contexte."
        + (f" Non supportées : {'; '.join(unsupported[:3])}" if unsupported else "")
    )
    return JudgeScore(score=faithful / total, rationale=rationale)


# ---------------------------------------------------------------------------
# Answer Relevancy — questions synthétiques + similarité cosinus (embeddings)
# (ragas: ResponseRelevancePrompt + embeddings.embed_query/embed_documents)
# ---------------------------------------------------------------------------

class _GeneratedQuestion(BaseModel):
    question: str
    noncommittal: int = Field(ge=0, le=1)


class _GeneratedQuestions(BaseModel):
    questions: list[_GeneratedQuestion]


_ANSWER_RELEVANCY_STRICTNESS = 3

_QUESTION_GENERATION_SYSTEM = (
    f"Génère {_ANSWER_RELEVANCY_STRICTNESS} questions différentes auxquelles "
    "la réponse suivante répondrait, et indique pour chacune si la réponse "
    "est évasive/vague/ambiguë (noncommittal=1, ex: \"information non "
    "trouvée\", \"je ne sais pas\") ou engagée sur le fond (noncommittal=0). "
    f"Réponds en JSON {{questions: [{{question, noncommittal}}, ...]}} avec "
    f"exactement {_ANSWER_RELEVANCY_STRICTNESS} entrées."
)


def answer_relevancy(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, answer: str, embedder) -> JudgeScore:
    prompt = f"Réponse :\n{answer}"
    generated = _call(llm_client, llm_config, _QUESTION_GENERATION_SYSTEM, prompt, _GeneratedQuestions).questions
    gen_questions = [g.question for g in generated if g.question]

    if not gen_questions:
        return JudgeScore(score=0.0, rationale="Le juge n'a généré aucune question synthétique exploitable.")

    all_noncommittal = all(g.noncommittal for g in generated)

    vectors = embedder.embed([question, *gen_questions]).dense
    question_vec, gen_vecs = vectors[0], vectors[1:]
    norms = np.linalg.norm(gen_vecs, axis=1) * np.linalg.norm(question_vec)
    norms[norms == 0] = 1e-10
    cosine_sim = (gen_vecs @ question_vec) / norms

    score = float(cosine_sim.mean()) * (0 if all_noncommittal else 1)
    rationale = (
        "Réponse évasive (noncommittal) sur toutes les questions générées — score forcé à 0."
        if all_noncommittal
        else f"Similarité cosinus moyenne question originale / {len(gen_questions)} questions synthétiques générées."
    )
    return JudgeScore(score=max(0.0, min(1.0, score)), rationale=rationale)


# ---------------------------------------------------------------------------
# Context Precision — Average Precision pondérée par le rang, verdicts
# d'utilité par chunk conditionnés à une réponse (ragas: ContextPrecisionPrompt)
# ---------------------------------------------------------------------------

class _Usefulness(BaseModel):
    reason: str
    verdict: int = Field(ge=0, le=1)


_CONTEXT_PRECISION_SYSTEM = (
    "Étant donné une question, une réponse et un passage documentaire, "
    "détermine si ce passage a été utile pour arriver à la réponse donnée. "
    "verdict=1 si utile, verdict=0 sinon. Réponds en JSON {reason, verdict}."
)


def context_precision(llm_client: BaseLLMClient, llm_config: LLMConfig, question: str, chunk_texts: list[str], answer: str) -> float:
    """Average Precision (spec §9) : verdicts d'utilité par chunk, jugés par
    rapport à `answer` (référence si disponible, sinon réponse générée) —
    pondérés par le rang du chunk dans le lot retrouvé."""
    if not chunk_texts:
        return 0.0

    verdicts = []
    for chunk_text in chunk_texts:
        prompt = f"Question : {question}\n\nRéponse : {answer}\n\nPassage :\n{chunk_text}"
        result = _call(llm_client, llm_config, _CONTEXT_PRECISION_SYSTEM, prompt, _Usefulness)
        verdicts.append(result.verdict)

    numerator = sum(
        (sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts))
    )
    denominator = sum(verdicts) + 1e-10
    return numerator / denominator if sum(verdicts) > 0 else 0.0


# ---------------------------------------------------------------------------
# Context Recall — attribution phrase par phrase de la réponse de référence
# (ragas: ContextRecallClassificationPrompt)
# ---------------------------------------------------------------------------

class _RecallStatement(BaseModel):
    statement: str
    reason: str
    attributed: int = Field(ge=0, le=1)


class _RecallClassifications(BaseModel):
    classifications: list[_RecallStatement]


_CONTEXT_RECALL_SYSTEM = (
    "Étant donné un contexte documentaire et une réponse de référence, "
    "analyse chaque phrase de la réponse et classe-la comme attribuable "
    "(1) ou non attribuable (0) au contexte donné. Réponds en JSON "
    "{classifications: [{statement, reason, attributed}, ...]}, une entrée "
    "par phrase de la réponse."
)


def context_recall(llm_client: BaseLLMClient, llm_config: LLMConfig, expected_answer: str, context_text: str) -> JudgeScore:
    """Context Recall (spec §9) — nécessite une vérité de référence écrite,
    seulement disponible sur le sous-ensemble de questions générées par
    template (voir data/eval/questions_generated_template.jsonl)."""
    prompt = f"Contexte documentaire :\n{context_text}\n\nRéponse de référence :\n{expected_answer}"
    classifications = _call(llm_client, llm_config, _CONTEXT_RECALL_SYSTEM, prompt, _RecallClassifications).classifications

    if not classifications:
        return JudgeScore(score=0.0, rationale="Aucune phrase classifiable dans la réponse de référence.")

    attributed = sum(1 for c in classifications if c.attributed)
    total = len(classifications)
    unattributed = [c.statement for c in classifications if not c.attributed]
    rationale = (
        f"{attributed}/{total} phrases de la réponse de référence attribuables au contexte."
        + (f" Non couvertes : {'; '.join(unattributed[:3])}" if unattributed else "")
    )
    return JudgeScore(score=attributed / total, rationale=rationale)
