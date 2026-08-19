#!/usr/bin/env python3
"""Phase 5 — Baselines D/E/F (spec §9) : génération, abstention, conflits.

Un seul passage par question à travers les composants Phase 4 (retrieve ->
sufficiency -> generate -> conflict -> abstention), en réutilisant
QueryPipeline._diff_explanation/_finalize_answer pour rester fidèle au vrai
comportement de production, mais en capturant la génération BRUTE (avant
abstention) séparément de la réponse finale (après abstention) — nécessaire
pour distinguer les métriques Baseline D (avant abstention) de Baseline E
(après abstention) à partir d'un seul run, sans dupliquer d'appels LLM.

Baseline D = C + génération/citations/sufficiency : RAGAS (Faithfulness,
             Answer Relevancy, Context Precision, Context Recall sur le
             sous-ensemble annoté) + Hallucination Rate, sur la génération brute.
Baseline E = D + abstention : mêmes métriques sur la réponse finale
             (après AbstentionGate) + taux d'abstention correcte/incorrecte.
Baseline F = E + détection de conflits : taux de conflits détectés sur les
             questions version_conflict (requires_obsolescence_check=True,
             où un conflit est attendu) vs sur les autres (où un conflit
             signalé est un faux positif).

Usage : --limit N pour un run partiel (smoke test avant le run complet).
"""

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# La console Windows utilise par défaut l'encodage OEM (cp1252/850, pas
# UTF-8) — print() plante sur certains caractères Unicode légitimes du
# corpus/des réponses générées (guillemets typographiques, tirets cadratins,
# etc.), faisant perdre une question entière en fin de traitement (trouvé
# en conditions réelles, pas une précaution ajoutée à l'aveugle).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.evaluation import ragas_eval
from src.evaluation.hallucination import hallucination_rate
from src.evaluation.reliability_metrics import citation_accuracy, sufficiency_precision
from src.generation.generator import Generator
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig
from src.llm.providers import ollama_client  # noqa: F401 — self-registers
from src.pipeline import QueryPipeline
from src.reliability.abstention import AbstentionGate
from src.reliability.conflict.detector import ConflictDetector
from src.reliability.sufficiency import SufficiencyChecker
from src.retrieval.hybrid_retriever import HybridRetriever

QUESTIONS_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_v1.jsonl"
OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "baseline_D_E_F_results.json"
RAW_OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "baseline_D_E_F_raw.jsonl"

CONFLICT_SUFFICIENCY_THRESHOLD = 0.7
RETRY_MAX_ATTEMPTS = 2  # timeout Ollama / échec de validation JSON du juge sont transitoires
RETRY_BACKOFF_SECONDS = 5


def build_components():
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
    retriever = HybridRetriever(store=store, embedder=embedder, use_reranker=True)
    retriever.warm_up()  # charge le reranker au démarrage, pas sur la première question chronométrée

    gen_config = LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,  # qwen2.5:3b-instruct — défaut projet
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        repeat_penalty=settings.llm_repeat_penalty,
    )
    judge_config = LLMConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct",
        base_url=settings.llm_base_url,
        temperature=0.1,
        # 300 tronquait le JSON du juge sur les verdicts qui listent des
        # claims (faithfulness, context_recall) pour les questions verboses
        # (conflit de version) — même symptôme que llm_max_tokens=2048 côté
        # générateur (settings.py). 1024 plutôt que 2048 : compromis constaté
        # en conditions réelles — 2048 en CPU (num_gpu=0, qwen2.5:7b-instruct)
        # faisait passer chaque question de ~5s à ~340s (5 appels juge par
        # question), un facteur ~70 jugé disproportionné pour le gain de
        # marge restant au-delà de 1024.
        max_tokens=1024,
        num_gpu=0,  # juge indépendant, CPU — évite la contention VRAM et le biais d'auto-évaluation
        repeat_penalty=settings.llm_repeat_penalty,
        # 120s (défaut LLMConfig) trop court depuis le passage de
        # context_precision à 10 chunks (#4) : un appel juge par chunk
        # (~13s/appel mesuré en CPU), et Ollama ralentit sous charge soutenue
        # (dérive observée sur le run complet précédent — questions de plus
        # en plus lentes après ~1h30). 300s laisse de la marge sans changer
        # le nombre de chunks évalués.
        timeout=300,
    )
    gen_client = LLMFactory.create(gen_config)
    judge_client = LLMFactory.create(judge_config)

    generator = Generator(llm_client=gen_client, llm_config=gen_config, store=store)
    # judge_client (7b) plutôt que gen_client (3b, défaut projet) : le 3b
    # mal-interprète des dumps JSON de spec pourtant sans ambiguïté (ex.
    # verdict "insufficient" sur un paramètre avec "required": true présent
    # tel quel dans le contexte, raison donnée fausse — "ne mentionne pas de
    # paramètres requis") — mesuré en conditions réelles, pas supposé. Limité
    # à ce script d'éval : la prod (src/api/dependencies.py) n'a qu'un seul
    # modèle chargé, lui ajouter un second changerait son empreinte réelle.
    sufficiency_checker = SufficiencyChecker(llm_client=judge_client)
    abstention_gate = AbstentionGate()
    conflict_detector = ConflictDetector(llm_client=gen_client, diff_dir=_PROJECT_ROOT / "data" / "diffs")

    return {
        "retriever": retriever,
        "generator": generator,
        "sufficiency_checker": sufficiency_checker,
        "abstention_gate": abstention_gate,
        "conflict_detector": conflict_detector,
        "gen_config": gen_config,
        "judge_client": judge_client,
        "judge_config": judge_config,
    }


def run_one_question(components: dict, q: dict) -> dict:
    retriever = components["retriever"]
    gen_config = components["gen_config"]
    settings = get_settings()

    t0 = time.perf_counter()
    retrieval = retriever.retrieve(
        query=q["question"], top_k=10, k_dense=settings.retrieval_k_dense, k_sparse=settings.retrieval_k_sparse
    )
    chunks = retrieval["results"]

    diff_explanation = QueryPipeline._diff_explanation(retrieval)
    sufficiency = components["sufficiency_checker"].check(
        q["question"], chunks, components["judge_config"], diff_explanation=diff_explanation
    )
    generation = components["generator"].generate(q["question"], chunks, diff_explanation=diff_explanation)

    conflicts = []
    if sufficiency.verdict != "insufficient" and sufficiency.confidence >= CONFLICT_SUFFICIENCY_THRESHOLD:
        conflicts = list(components["conflict_detector"].detect_textual(chunks, gen_config))
        # Ce script réimplémente sa propre chaîne retrieve->sufficiency->
        # generate->conflict (pour capturer séparément la génération brute et
        # finale depuis un seul passage, voir docstring du module) plutôt que
        # de réutiliser QueryPipeline._detect_conflicts() — doit donc recevoir
        # le même fix (audit #12) séparément, sinon les conflits structurels
        # restent invisibles ici même une fois QueryPipeline corrigé.
        diff_report = retrieval.get("diff_report")
        if diff_report:
            structural = components["conflict_detector"].detect_structural(
                diff_report["source_id"], diff_report["version_from"], diff_report["version_to"]
            )
            if structural.conflict:
                conflicts.append(structural)

    reranker_scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
    decision = components["abstention_gate"].evaluate(
        reranker_scores=reranker_scores,
        generation_confidence=generation.confidence,
        sufficiency_verdict=sufficiency,
        planner_confidence=retrieval.get("confidence"),
    )
    final_answer, final_citations = QueryPipeline._finalize_answer(decision, generation)
    latency_ms = (time.perf_counter() - t0) * 1000

    # diff_explanation inclus : c'est une partie réelle du contexte que le
    # générateur a reçu (voir build_user_message) — sans lui, toute
    # affirmation sourcée depuis le diff plutôt qu'un chunk brut était jugée
    # "non supportée" par erreur (mesuré : Faithfulness 0.13 sur
    # version_conflict vs 0.61 sur factual, largement artificiel).
    context_text = "\n---\n".join(c.get("text", "") for c in chunks)
    if diff_explanation:
        context_text += "\n---\n" + diff_explanation
    # Les 10 chunks réellement utilisés pour la génération, pas seulement les
    # 3 premiers — le plafond initial (coût du juge 7b CPU) sous-estimait la
    # précision réelle du contexte fourni au générateur.
    chunk_texts = [c.get("text", "") for c in chunks]

    judge_client, judge_config = components["judge_client"], components["judge_config"]
    precision_answer = q.get("expected_answer") or generation.answer
    faith = ragas_eval.faithfulness(judge_client, judge_config, q["question"], context_text, generation.answer)
    relevancy = ragas_eval.answer_relevancy(judge_client, judge_config, q["question"], generation.answer, components["retriever"].embedder)
    ctx_precision = ragas_eval.context_precision(judge_client, judge_config, q["question"], chunk_texts, precision_answer)
    ctx_recall = None
    if q.get("expected_answer"):
        ctx_recall = ragas_eval.context_recall(judge_client, judge_config, q["expected_answer"], context_text).score
    cit_accuracy = citation_accuracy(judge_client, judge_config, generation.answer, generation.citations)

    return {
        "question": q["question"],
        "category": q["category"],
        "requires_obsolescence_check": q.get("requires_obsolescence_check", False),
        "expected_sources": q.get("expected_sources", []),
        # Calculé par QueryPlanner à chaque requête (voir hybrid_retriever.py)
        # mais jamais persisté jusqu'ici — aucune comparaison possible contre
        # `category` (vérité terrain) sans le garder. Pas le même vocabulaire
        # (intent = comment router la requête, category = type de question
        # du jeu de test) : rapprochement valide seulement pour factual↔factual,
        # version_conflict↔comparative, ambiguous↔ambiguous — multi_source et
        # abstention n'ont pas de correspondance attendue unique.
        "planner_intent": retrieval.get("planner_intent"),
        "raw_answer": generation.answer,
        "raw_citations": [c.model_dump(mode="json") for c in generation.citations],
        "final_answer": final_answer,
        "final_citations": [c.model_dump(mode="json") for c in final_citations],
        "zone": decision.zone,
        "action": decision.action,
        "sufficiency_verdict": sufficiency.verdict,
        "n_conflicts": len(conflicts),
        "conflict_types": [c.type for c in conflicts] if conflicts else [],
        "faithfulness": faith.score,
        "answer_relevancy": relevancy.score,
        "context_precision": ctx_precision,
        "context_recall": ctx_recall,
        "citation_accuracy": cit_accuracy["citation_accuracy"],
        "should_abstain": q["category"] == "abstention",
        "latency_ms": latency_ms,
    }


def run_one_question_with_retry(components: dict, q: dict) -> dict:
    """Retry borné sur erreurs transitoires (timeout Ollama, échec de
    validation JSON du juge) — auparavant une seule erreur perdait
    silencieusement la question (8/50 dans l'évaluation Phase 5 d'origine),
    sans distinction entre bug réel et aléa réseau/inférence."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return run_one_question(components, q)
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS:
                print(f"  -> échec (tentative {attempt}/{RETRY_MAX_ATTEMPTS}, {exc}), retry dans {RETRY_BACKOFF_SECONDS}s...")
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Nombre de questions à traiter (smoke test)")
    parser.add_argument("--questions-path", type=Path, default=None, help="Fichier de questions alternatif (JSONL)")
    args = parser.parse_args()

    questions_path = args.questions_path or QUESTIONS_PATH
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Traitement de {len(questions)} questions (source: {questions_path.name})...")

    components = build_components()

    records = []
    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUT_PATH, "w", encoding="utf-8") as raw_f:
        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            try:
                record = run_one_question_with_retry(components, q)
            except Exception as exc:
                print(f"[{i}/{len(questions)}] ERREUR (après {RETRY_MAX_ATTEMPTS} tentatives) sur '{q['question'][:60]}': {exc}")
                # Cause persistée (pas seulement affichée sur stdout, perdu à
                # la fin du run) : sans ça, un échec après retry ne laisse
                # aucune trace exploitable pour diagnostiquer sa cause réelle
                # après coup (constaté sur les 19/50 questions perdues de
                # l'évaluation précédente — cause jamais reconstituable).
                raw_f.write(json.dumps(
                    {"question": q["question"], "category": q.get("category"),
                     "error": f"{type(exc).__name__}: {exc}", "attempts": RETRY_MAX_ATTEMPTS},
                    ensure_ascii=False,
                ) + "\n")
                raw_f.flush()
                continue
            records.append(record)
            raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_f.flush()
            dt = time.perf_counter() - t0
            print(f"[{i}/{len(questions)}] ({dt:.1f}s) {q['category']:<18} zone={record['zone']:<10} "
                  f"faith={record['faithfulness']:.2f} conflicts={record['n_conflicts']} -> {q['question'][:70]}")

    # ---- Agrégation par baseline ----
    def avg(key, subset=None):
        vals = [r[key] for r in (subset or records) if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    # Hallucination Rate — objets simplifiés compatibles avec le protocole attendu.
    # `faithfulness` vient du juge indépendant (ragas_eval.faithfulness, déjà
    # calculé par question ci-dessus), pas d'une auto-déclaration du générateur.
    class _R:
        def __init__(self, citations, faithfulness):
            self.citations = citations
            self.faithfulness = faithfulness

    def halluc(citations_key):
        objs = [_R(r[citations_key], r["faithfulness"]) for r in records]
        return hallucination_rate(objs)

    # Sufficiency Precision (spec §15.3) — indépendante de D/E/F (porte sur
    # le verdict de SufficiencyChecker, calculé avant génération/abstention).
    suff_precision = sufficiency_precision(records)

    baseline_d = {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "citation_accuracy": avg("citation_accuracy"),
        "sufficiency_precision": suff_precision["sufficiency_precision"],
        "hallucination": halluc("raw_citations"),
    }

    abstention_qs = [r for r in records if r["category"] == "abstention"]
    non_abstention_qs = [r for r in records if r["category"] != "abstention"]
    correct_abstentions = sum(1 for r in abstention_qs if r["action"] == "abstain")
    incorrect_abstentions = sum(1 for r in non_abstention_qs if r["action"] == "abstain")
    baseline_e = {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "citation_accuracy": avg("citation_accuracy"),
        "sufficiency_precision": suff_precision["sufficiency_precision"],
        "hallucination": halluc("final_citations"),
        "correct_abstention_rate": (correct_abstentions / len(abstention_qs)) if abstention_qs else None,
        "false_abstention_rate": (incorrect_abstentions / len(non_abstention_qs)) if non_abstention_qs else None,
        "n_abstention_questions": len(abstention_qs),
    }

    version_conflict_qs = [r for r in records if r["requires_obsolescence_check"]]
    other_qs = [r for r in records if not r["requires_obsolescence_check"]]
    missed_conflicts = sum(1 for r in version_conflict_qs if r["n_conflicts"] == 0)
    false_conflicts = sum(1 for r in other_qs if r["n_conflicts"] > 0)
    baseline_f = {
        **baseline_e,
        "missed_conflict_rate": (missed_conflicts / len(version_conflict_qs)) if version_conflict_qs else None,
        "false_conflict_rate": (false_conflicts / len(other_qs)) if other_qs else None,
        "n_version_conflict_questions": len(version_conflict_qs),
    }

    results = {
        "D_generation_citations_sufficiency": baseline_d,
        "E_plus_abstention": baseline_e,
        "F_plus_conflict_detection": baseline_f,
        "_meta": {"n_questions": len(records), "n_requested": len(questions)},
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Résumé ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRésultats agrégés : {OUT_PATH}")
    print(f"Détail par question : {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
