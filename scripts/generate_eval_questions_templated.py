#!/usr/bin/env python3
"""Phase 5 — génère les questions "factuelles" et "conflit de versions" par
template, depuis les vraies données du corpus (specs OpenAPI Stripe/Binance,
data/diffs/*.json), sans écriture manuelle.

expected_answer / expected_sources dérivent directement de données
déterministes (le diff exact, ou la spec OpenAPI parsée) — vérité de
référence garantie exacte, utilisable pour Context Recall (RAGAS) sur ce
sous-ensemble.

Sortie : data/eval/questions_generated_template.jsonl (candidat — audit
manuel requis avant fusion dans questions_v1.jsonl, voir spec §9).
"""

import json
import re
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DIFFS_DIR = _PROJECT_ROOT / "data" / "diffs"
STRIPE_DIR = _PROJECT_ROOT / "raw" / "specs-api" / "stripe"
BINANCE_DIR = _PROJECT_ROOT / "raw" / "specs-api" / "binance"
OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_generated_template.jsonl"


def _posix(p: Path) -> str:
    return p.relative_to(_PROJECT_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Catégorie "conflit de versions" — depuis data/diffs/*.json (déterministe)
# ---------------------------------------------------------------------------

def _change_score(field: str, old: str, new: str) -> int:
    """Priorise les changements substantiels (pas juste une différence de
    casse) et le champ markdown (contient la description complète, pas
    juste le titre)."""
    if old.strip().lower() == new.strip().lower():
        return 0
    score = 1
    if field == "markdown":
        score += 3
    score += min(len(old) + len(new), 400) // 100
    return score


def collect_version_conflict_questions(n: int = 10) -> list[dict]:
    candidates = []
    for diff_path in sorted(DIFFS_DIR.glob("*.json")):
        diff = json.loads(diff_path.read_text(encoding="utf-8"))
        v_from, v_to = diff["version_from"], diff["version_to"]
        for change in diff["changes"]:
            if change["change_type"] != "modified":
                continue
            field_changes = change.get("field_changes") or {}
            best_field, best_score = None, 0
            for field, vals in field_changes.items():
                old, new = vals.get("old"), vals.get("new")
                if not old or not new:
                    continue
                s = _change_score(field, old, new)
                if s > best_score:
                    best_field, best_score = field, s
            if best_field is None:
                continue
            candidates.append(
                {
                    "score": best_score,
                    "key": change["key"],
                    "version_from": v_from,
                    "version_to": v_to,
                    "field": best_field,
                    "old": field_changes[best_field]["old"],
                    "new": field_changes[best_field]["new"],
                }
            )

    # Dédoublonner par (clé, paire de versions) — un même endpoint qui
    # change entre plusieurs paires de versions produit des questions
    # légitimement différentes, on ne garde qu'un doublon exact.
    best_per_key: dict[tuple, dict] = {}
    for c in candidates:
        dedup_key = (c["key"], c["version_from"], c["version_to"])
        if dedup_key not in best_per_key or c["score"] > best_per_key[dedup_key]["score"]:
            best_per_key[dedup_key] = c
    # Diversité : au plus 3 questions par paire de versions, pour ne pas
    # laisser une seule paire (souvent legacy->v2213, la plus riche en
    # changements) monopoliser la catégorie.
    ranked_all = sorted(best_per_key.values(), key=lambda c: -c["score"])
    ranked, per_pair_count = [], {}
    for c in ranked_all:
        pair = (c["version_from"], c["version_to"])
        if per_pair_count.get(pair, 0) >= 3:
            continue
        ranked.append(c)
        per_pair_count[pair] = per_pair_count.get(pair, 0) + 1
        if len(ranked) >= n:
            break

    questions = []
    for c in ranked:
        v_from_path = _posix(STRIPE_DIR / f"spec3-{c['version_from']}.yaml")
        v_to_path = _posix(STRIPE_DIR / f"spec3-{c['version_to']}.yaml")
        field_label = {"markdown": "la description", "summary": "le résumé", "text": "le texte"}.get(c["field"], c["field"])
        question = (
            f"Qu'est-ce qui a changé dans {field_label} de l'endpoint "
            f"« {c['key']} » entre la version {c['version_from']} et la version {c['version_to']} "
            f"de l'API Stripe ?"
        )
        expected_answer = (
            f"Version {c['version_from']} : {c['old'].strip()}\n\n"
            f"Version {c['version_to']} : {c['new'].strip()}"
        )
        questions.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": [v_from_path, v_to_path],
                "category": "version_conflict",
                "eval_criteria": "citation_exacte_deux_versions",
                "requires_obsolescence_check": True,
                "_generation": {"method": "template_diff", "key": c["key"], "field": c["field"]},
            }
        )
    return questions


# ---------------------------------------------------------------------------
# Catégorie "factuelles" — depuis les vraies specs OpenAPI (déterministe)
# ---------------------------------------------------------------------------

def _load_openapi(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_endpoints(spec: dict, source_path: Path):
    for path_str, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue
            yield path_str, method, op, source_path


def collect_factual_questions(n: int = 10) -> list[dict]:
    sources = []
    # Version la plus récente de Stripe (v2323) — évite d'interroger un
    # comportement obsolète pour une question "factuelle" simple.
    stripe_latest = STRIPE_DIR / "spec3-v2323.yaml"
    if stripe_latest.exists():
        sources.append(_load_openapi(stripe_latest), )
        stripe_spec = sources[-1]
    binance_spec_path = BINANCE_DIR / "spot_api.yaml"
    binance_spec = _load_openapi(binance_spec_path) if binance_spec_path.exists() else None

    candidates = []
    for spec, src_path in ((stripe_spec, stripe_latest), (binance_spec, binance_spec_path)):
        if spec is None:
            continue
        for path_str, method, op, source_path in _iter_endpoints(spec, src_path):
            params = [p for p in op.get("parameters", []) if isinstance(p, dict)]
            # Exclure les paramètres de chemin (`in: path`) — trivialement
            # toujours requis par construction REST, ça ne teste pas une
            # vraie récupération d'information dans le corpus.
            required = [p for p in params if p.get("required") and p.get("in") != "path"]
            if not required:
                continue
            candidates.append((path_str, method, op, source_path, required))

    # Répartir équitablement entre les deux sources disponibles.
    by_source: dict[str, list] = {}
    for c in candidates:
        by_source.setdefault(_posix(c[3]), []).append(c)

    picked = []
    sources_cycle = list(by_source.keys())
    i = 0
    while len(picked) < n and any(by_source.values()):
        src = sources_cycle[i % len(sources_cycle)]
        if by_source[src]:
            picked.append(by_source[src].pop(0))
        i += 1
        if i > 10 * n:
            break

    questions = []
    for path_str, method, op, source_path, required in picked:
        param_descriptions = []
        for p in required:
            name = p.get("name", "unknown")
            ptype = p.get("schema", {}).get("type", p.get("type", "unknown"))
            param_descriptions.append(f"{name} ({ptype})")
        question = f"Quel(s) paramètre(s) sont requis pour {method.upper()} {path_str} ?"
        expected_answer = "Paramètre(s) requis : " + ", ".join(param_descriptions)
        questions.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": [_posix(source_path)],
                "category": "factual",
                "eval_criteria": "citation_exacte",
                "requires_obsolescence_check": False,
                "_generation": {"method": "template_openapi", "path": path_str, "method_http": method},
            }
        )
    return questions


def main():
    version_conflict_qs = collect_version_conflict_questions(15)
    factual_qs = collect_factual_questions(15)
    all_qs = factual_qs + version_conflict_qs

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q in all_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Générées : {len(factual_qs)} factuelles, {len(version_conflict_qs)} conflit_versions")
    print(f"Écrit dans {OUT_PATH}")


if __name__ == "__main__":
    main()
