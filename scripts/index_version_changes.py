#!/usr/bin/env python3
"""Indexe les changements substantiels des diffs précalculés (data/diffs/)
comme des nœuds "Change" recherchables sémantiquement dans Qdrant — chemin
de retrieval dédié aux questions comparatives (intent="comparative"),
inspiré de VersionRAG (arXiv:2510.08109 §4.1) : les changements y sont des
objets de première classe, indexés et recherchables au même titre que le
contenu, pas seulement un texte généré à la volée au moment de la requête
(voir src/pipeline.py::_diff_explanation, qui reste le mécanisme utilisé
pour les autres questions comparatives sans nœud Change correspondant).

Génère une description sémantique par LLM (pas un template mécanique) avec
le modèle FORT du projet (qwen2.5:7b-instruct, CPU), jamais le modèle
rapide utilisé en génération live — l'étude d'ablation de VersionRAG montre
que la qualité de l'indexation borne la qualité finale ("indexing quality
sets an upper bound on achievable accuracy", §6.5). Coût ponctuel hors
ligne, aucun impact sur la latence des requêtes live.

Filtre aux changements substantiels (_change_score, même logique que
generate_eval_questions_templated.py) — décrire par LLM les ~900 changements
bruts d'un diff Plaid complet serait disproportionné pour le bruit que ça
ajouterait (différences de casse, reformatage).

Écrit des points Qdrant marqués `node_type="change"`, jamais retournés par
le retrieval générique (voir QdrantStore._exclude_change_nodes) — seul
HybridRetriever._search_change_nodes (intent="comparative") les interroge.

Usage : python scripts/index_version_changes.py [--source-id plaid] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embeddings.bge_m3 import BGEEmbedder
from src.embeddings.vector_store import QdrantStore
from src.ingestion.version_diff import Change, DiffReport
from src.llm.factory import LLMFactory
from src.llm.interface import LLMConfig, LLMMessage
from src.llm.providers import ollama_client  # noqa: F401 — self-registers

DIFFS_DIR = _PROJECT_ROOT / "data" / "diffs"

_DESCRIPTION_SYSTEM_PROMPT = (
    "Tu décris un changement technique entre deux versions d'une "
    "spécification API, en une phrase factuelle et précise, sans jugement "
    "ni supposition, sans reformuler la question. Réponds uniquement avec "
    "la description, rien d'autre."
)


def _change_score(change: Change) -> int:
    """Priorise les changements substantiels — même logique que
    generate_eval_questions_templated.py::_change_score, réutilisée ici pour
    ne décrire par LLM (coûteux) que les changements qui en valent la peine."""
    if change.change_type in ("added", "removed"):
        return 2
    field_changes = change.field_changes or {}
    score = 0
    for field, vals in field_changes.items():
        old, new = vals.get("old"), vals.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        if old.strip().lower() == new.strip().lower():
            continue
        score += 3 if field in ("markdown", "text", "description") else 1
    return score


def _describe_change(llm_client, llm_config: LLMConfig, diff_report: DiffReport, change: Change) -> str:
    fields = (
        ", ".join(f"{f}: {v.get('old')!r} -> {v.get('new')!r}" for f, v in change.field_changes.items())
        if change.field_changes
        else ""
    )
    user = (
        f"Source: {diff_report.source_id}, version {diff_report.version_from} -> {diff_report.version_to}\n"
        f"Élément: {change.key}\n"
        f"Type de changement: {change.change_type}\n" + (f"Champs modifiés: {fields}" if fields else "")
    )
    response = llm_client.complete(
        [
            LLMMessage(role="system", content=_DESCRIPTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ],
        llm_config,
    )
    return response.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-id", default=None, help="Ne traiter que ce source_id (sinon : tous les diffs présents)")
    parser.add_argument("--pair", default=None, help="Ne traiter que cette paire de versions précise, format 'from:to' (ex. '1.5.0-beta:1.8.1-beta') — nécessite --source-id")
    parser.add_argument("--min-score", type=int, default=2, help="Score minimal pour indexer un changement")
    parser.add_argument("--limit", type=int, default=None, help="Nombre max de changements à traiter au total (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="N'écrit rien dans Qdrant")
    args = parser.parse_args()

    settings = get_settings()
    judge_config = LLMConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct",
        base_url=settings.llm_base_url,
        temperature=0.0,
        max_tokens=200,
        num_gpu=0,  # modèle fort, CPU — cohérent avec le reste du projet (juge RAGAS)
        timeout=120,
    )
    # dry-run : aucun appel coûteux (LLM, chargement embedder) — juste le
    # comptage des changements qualifiants, pour évaluer le coût avant de
    # lancer pour de vrai.
    llm_client = None if args.dry_run else LLMFactory.create(judge_config)

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    embedder = None
    if not args.dry_run:
        embedder = BGEEmbedder(
            model_name=settings.embedding_model,
            dtype=settings.embedding_dtype,
            backend=settings.embedding_backend,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )

    if args.pair:
        if not args.source_id:
            raise SystemExit("--pair nécessite --source-id")
        v_from, v_to = args.pair.split(":", 1)
        pattern = f"{args.source_id}_{v_from}_{v_to}.json"
    else:
        pattern = f"{args.source_id}_*.json" if args.source_id else "*.json"
    diff_paths = sorted(DIFFS_DIR.glob(pattern))
    if not diff_paths:
        raise SystemExit(f"Aucun diff trouvé sous {DIFFS_DIR} (pattern={pattern})")

    total_indexed = 0
    remaining = args.limit
    for diff_path in diff_paths:
        if remaining is not None and remaining <= 0:
            break
        diff_report = DiffReport(**json.loads(diff_path.read_text(encoding="utf-8")))
        qualifying = [c for c in diff_report.changes if _change_score(c) >= args.min_score]
        if remaining is not None:
            qualifying = qualifying[:remaining]
        print(f"{diff_path.name}: {len(qualifying)}/{len(diff_report.changes)} changements substantiels (score >= {args.min_score})")

        if args.dry_run:
            for change in qualifying:
                hierarchy_path = change.key.rsplit("|", 1)[0]
                print(f"  [{change.change_type}] {hierarchy_path[:90]} (score={_change_score(change)})")
            total_indexed += len(qualifying)
            if remaining is not None:
                remaining -= len(qualifying)
            continue

        nodes: list[dict] = []
        texts: list[str] = []
        for change in qualifying:
            hierarchy_path = change.key.rsplit("|", 1)[0]
            description = _describe_change(llm_client, judge_config, diff_report, change)
            node_id = f"change-{diff_report.source_id}-{diff_report.version_from}-{diff_report.version_to}-{change.key}"
            nodes.append(
                {
                    "chunk_id": node_id,
                    "text": description,
                    "source_id": diff_report.source_id,
                    "version_from": diff_report.version_from,
                    "version_to": diff_report.version_to,
                    "hierarchy_path": hierarchy_path,
                    "change_type": change.change_type,
                }
            )
            texts.append(description)
            print(f"  [{change.change_type}] {hierarchy_path[:70]} -> {description[:100]}")

        if nodes:
            embeddings = embedder.embed(texts).dense
            store.upsert_change_nodes(nodes, embeddings)
        total_indexed += len(nodes)
        if remaining is not None:
            remaining -= len(qualifying)

    suffix = " (dry-run, rien écrit dans Qdrant)" if args.dry_run else ""
    print(f"\n{total_indexed} nœuds Change indexés{suffix}")


if __name__ == "__main__":
    main()
