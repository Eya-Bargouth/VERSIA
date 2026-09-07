#!/usr/bin/env python3
"""Phase 5 — génère les questions "factuelles" et "conflit de versions" par
template, depuis les vraies données du corpus (specs OpenAPI découvertes
automatiquement sous raw/, data/diffs/*.json), sans écriture manuelle.

Générique par construction : les sources sont découvertes via
`discover_sources` (comme `generate_version_diffs.py`), pas codées en dur.
Une source "factuelle" est repérée par sa FORME (un spec REST-like : clé
`paths` de type dict au niveau racine), pas par son nom. Ajouter/retirer une
source ne demande aucune modification de ce script.

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

from src.config.source_config import SourceConfig
from src.ingestion.pipeline import discover_sources

DIFFS_DIR = _PROJECT_ROOT / "data" / "diffs"
RAW_DIR = _PROJECT_ROOT / "raw"
VERSIONING_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "versioning"
OUT_PATH = _PROJECT_ROOT / "data" / "eval" / "questions_generated_template.jsonl"


def _posix(p: Path) -> str:
    return p.relative_to(_PROJECT_ROOT).as_posix()


def _load_structured(path: Path):
    """Charge un fichier YAML/JSON en structure Python, ou None si ce n'est
    pas parsable comme tel (les sources factuelles ne sont jamais du
    Markdown/PDF)."""
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def _version_file_map(source: SourceConfig) -> dict[str, Path]:
    """tag de version -> chemin fichier, dérivé du `version_pattern`
    générique de la source (voir tests/fixtures/versioning/<source_id>.yaml)
    plutôt que d'un format de nom de fichier codé en dur."""
    mapping: dict[str, Path] = {}
    if source.version_pattern is None or source.source_dir is None:
        return mapping
    pattern = re.compile(source.version_pattern)
    for f in sorted(source.source_dir.iterdir()):
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            mapping[m.group("version")] = f
    return mapping


def _is_rest_like_spec(data) -> bool:
    """Signature générique d'un spec REST-like : une clé `paths` dont la
    valeur est un dict non vide — propriété de FORME (OpenAPI/Swagger),
    jamais un nom de source particulier."""
    return isinstance(data, dict) and isinstance(data.get("paths"), dict) and bool(data.get("paths"))


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


def collect_version_conflict_questions(sources: list[SourceConfig], n: int = 15) -> list[dict]:
    version_maps = {s.source_id: _version_file_map(s) for s in sources if s.version_pattern}

    candidates = []
    for diff_path in sorted(DIFFS_DIR.glob("*.json")):
        diff = json.loads(diff_path.read_text(encoding="utf-8"))
        source_id = diff.get("source_id")
        if source_id not in version_maps:
            continue
        v_from, v_to = diff["version_from"], diff["version_to"]
        for change in diff["changes"]:
            if change["change_type"] != "modified":
                continue
            field_changes = change.get("field_changes") or {}
            best_field, best_score = None, 0
            for field, vals in field_changes.items():
                old, new = vals.get("old"), vals.get("new")
                # Seuls les champs textuels produisent une question/réponse
                # lisible en langage naturel (ex. "keys" — liste des clés
                # d'un dict — change aussi entre versions mais ne s'y prête
                # pas).
                if not isinstance(old, str) or not isinstance(new, str) or not old or not new:
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
                    "source_id": source_id,
                    "version_from": v_from,
                    "version_to": v_to,
                    "field": best_field,
                    "old": field_changes[best_field]["old"],
                    "new": field_changes[best_field]["new"],
                }
            )

    # Dédoublonner par (source, clé, paire de versions) — un même endpoint
    # qui change entre plusieurs paires de versions produit des questions
    # légitimement différentes, on ne garde qu'un doublon exact.
    best_per_key: dict[tuple, dict] = {}
    for c in candidates:
        dedup_key = (c["source_id"], c["key"], c["version_from"], c["version_to"])
        if dedup_key not in best_per_key or c["score"] > best_per_key[dedup_key]["score"]:
            best_per_key[dedup_key] = c
    # Diversité : au plus 3 questions par (source, paire de versions), pour
    # ne pas laisser une seule paire monopoliser la catégorie.
    ranked_all = sorted(best_per_key.values(), key=lambda c: -c["score"])
    ranked, per_pair_count = [], {}
    for c in ranked_all:
        pair = (c["source_id"], c["version_from"], c["version_to"])
        if per_pair_count.get(pair, 0) >= 3:
            continue
        ranked.append(c)
        per_pair_count[pair] = per_pair_count.get(pair, 0) + 1
        if len(ranked) >= n:
            break

    questions = []
    for c in ranked:
        vmap = version_maps.get(c["source_id"], {})
        from_path, to_path = vmap.get(c["version_from"]), vmap.get(c["version_to"])
        if from_path is None or to_path is None:
            continue
        field_label = {"markdown": "la description", "summary": "le résumé", "text": "le texte"}.get(c["field"], c["field"])
        question = (
            f"Qu'est-ce qui a changé dans {field_label} de « {c['key']} » "
            f"entre la version {c['version_from']} et la version {c['version_to']} "
            f"de la source « {c['source_id']} » ?"
        )
        expected_answer = (
            f"Version {c['version_from']} : {c['old'].strip()}\n\n"
            f"Version {c['version_to']} : {c['new'].strip()}"
        )
        questions.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": [_posix(from_path), _posix(to_path)],
                "category": "version_conflict",
                "eval_criteria": "citation_exacte_deux_versions",
                "requires_obsolescence_check": True,
                "_generation": {"method": "template_diff", "source_id": c["source_id"], "key": c["key"], "field": c["field"]},
            }
        )
    return questions


# ---------------------------------------------------------------------------
# Catégorie "factuelles" — depuis les vraies specs REST-like (déterministe)
# ---------------------------------------------------------------------------

def _iter_endpoints(spec: dict, source_path: Path):
    for path_str, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue
            yield path_str, method, op, source_path


def _factual_reference_file(source: SourceConfig) -> Path | None:
    """Le fichier le plus pertinent d'une source pour des questions
    factuelles simples : la dernière version si versionnée (évite
    d'interroger un comportement obsolète), sinon le premier fichier trouvé
    dans le dossier."""
    if source.source_dir is None or not source.source_dir.exists():
        return None
    if source.version_pattern and source.version_order:
        vmap = _version_file_map(source)
        return vmap.get(source.version_order[-1])
    files = sorted(p for p in source.source_dir.iterdir() if p.is_file())
    return files[0] if files else None


def _resolve_ref(spec: dict, node):
    """Résout un {"$ref": "#/a/b/c"} (JSON Pointer interne, RFC 6901) contre
    la racine du document — un seul niveau, suffisant pour un schema de
    requestBody qui pointe directement vers sa définition dans
    components/schemas (forme standard OpenAPI, générique)."""
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    target = spec
    try:
        for part in ref[2:].split("/"):
            target = target[part]
    except (KeyError, TypeError):
        return node
    return target


def _required_params(op: dict, spec: dict) -> list[dict]:
    """Paramètres requis d'une opération, uniformisés en {"name", "type"} —
    couvre les deux formes génériques OpenAPI pour exprimer un paramètre
    requis (propriété de FORME du format, pas de sémantique d'un corpus) :
    - `parameters: [{name, required: true, in: !=path, schema: {type}}]`
      (style query/header — ex. Stripe, Binance) ;
    - `requestBody.content.<media>.schema.required: [str, ...]` avec le
      type dans `schema.properties.<name>.type` (style POST+JSON body — ex.
      Plaid, où la quasi-totalité des endpoints n'utilisent jamais
      `parameters`)."""
    result: list[dict] = []

    params = [p for p in op.get("parameters", []) if isinstance(p, dict)]
    # Exclure les paramètres de chemin (`in: path`) — trivialement toujours
    # requis par construction REST, ça ne teste pas une vraie récupération
    # d'information dans le corpus.
    for p in params:
        if p.get("required") and p.get("in") != "path":
            name = p.get("name", "unknown")
            ptype = p.get("schema", {}).get("type", p.get("type", "unknown"))
            result.append({"name": name, "type": ptype})

    request_body = op.get("requestBody")
    if isinstance(request_body, dict):
        for media in (request_body.get("content") or {}).values():
            schema = media.get("schema") if isinstance(media, dict) else None
            schema = _resolve_ref(spec, schema)
            if not isinstance(schema, dict):
                continue
            required_names = schema.get("required")
            if not isinstance(required_names, list):
                continue
            properties = schema.get("properties") or {}
            for name in required_names:
                if not isinstance(name, str):
                    continue
                prop = properties.get(name) if isinstance(properties, dict) else None
                prop = _resolve_ref(spec, prop)
                ptype = prop.get("type", "unknown") if isinstance(prop, dict) else "unknown"
                result.append({"name": name, "type": ptype})
            break  # un seul media type suffit — required est identique entre eux en pratique
    return result


def collect_factual_questions(sources: list[SourceConfig], n: int = 15) -> list[dict]:
    rest_sources = []
    for source in sources:
        ref_file = _factual_reference_file(source)
        if ref_file is None:
            continue
        data = _load_structured(ref_file)
        if _is_rest_like_spec(data):
            rest_sources.append((data, ref_file))

    candidates = []
    for spec, src_path in rest_sources:
        for path_str, method, op, source_path in _iter_endpoints(spec, src_path):
            required = _required_params(op, spec)
            if not required:
                continue
            candidates.append((path_str, method, op, source_path, required))

    # Répartir équitablement entre toutes les sources REST-like disponibles.
    by_source: dict[str, list] = {}
    for c in candidates:
        by_source.setdefault(_posix(c[3]), []).append(c)

    picked = []
    sources_cycle = list(by_source.keys())
    i = 0
    while len(picked) < n and any(by_source.values()) and sources_cycle:
        src = sources_cycle[i % len(sources_cycle)]
        if by_source[src]:
            picked.append(by_source[src].pop(0))
        i += 1
        if i > 10 * n:
            break

    questions = []
    for path_str, method, op, source_path, required in picked:
        param_descriptions = [f"{p['name']} ({p['type']})" for p in required]
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
    sources = discover_sources(RAW_DIR, versioning_dir=VERSIONING_DIR)

    version_conflict_qs = collect_version_conflict_questions(sources, 15)
    factual_qs = collect_factual_questions(sources, 15)
    all_qs = factual_qs + version_conflict_qs

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q in all_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Générées : {len(factual_qs)} factuelles, {len(version_conflict_qs)} conflit_versions")
    print(f"Écrit dans {OUT_PATH}")


if __name__ == "__main__":
    main()
