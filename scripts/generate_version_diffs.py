#!/usr/bin/env python3
"""Calcule les diffs entre toutes les paires de versions de chaque source
versionnée découverte dans manifest_dir, et les persiste dans data/diffs/ via
VersionDiffEngine.

Générique par construction : découvre tous les manifestes déclarant
``versioning.strategy != "none"`` (peu importe le parser — yaml_structured,
markdown, json, docling — ou le source_id), sans aucun nom de corpus codé en
dur. Ajouter une nouvelle source versionnée ne demande aucune modification de
ce script, seulement un manifeste avec ``versioning.pattern``/``.order``
déclarés (voir tests/fixtures/manifests/stripe.yaml pour un exemple).

Aucun builder DOM ne peuple actuellement DOMNode.version_tag depuis
manifest.versioning (gap pré-existant, même famille que le status=None
documenté en Phase 3 — voir docs/PHASE_3_SUMMARY.md §6). VersionDiffEngine.diff()
se rabat alors sur des tags génériques "v1"/"v2" pour toutes les paires, ce qui
fait collisionner les fichiers de sortie. Ce script contourne le problème
localement (sans toucher aux builders partagés) en dérivant le tag depuis
``manifest.versioning.pattern`` appliqué au nom de fichier.
"""
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.manifest_schema import SourceManifest
from src.dom.models import DocumentTree
from src.ingestion.pipeline import discover_manifests, select_builder
from src.ingestion.version_diff import VersionDiffEngine


def _resolve_files(raw_dir: Path, manifest: SourceManifest) -> list[Path]:
    base_dir = manifest.scope.get("base_dir")
    source_dir = raw_dir / base_dir if base_dir else raw_dir
    files: set[Path] = set()
    for pattern in manifest.scope.get("include", []):
        files.update(source_dir.glob(pattern))
    for pattern in manifest.scope.get("exclude", []):
        files -= set(source_dir.glob(pattern))
    return sorted(files)


def _extract_version_tag(file_path: Path, manifest: SourceManifest) -> str:
    if manifest.versioning.strategy != "filename_pattern" or not manifest.versioning.pattern:
        raise ValueError(
            f"manifest {manifest.source_id} n'utilise pas versioning.strategy=filename_pattern "
            "(seule stratégie supportée par ce script pour l'instant)"
        )
    match = re.search(manifest.versioning.pattern, file_path.name)
    if not match:
        raise ValueError(f"{file_path.name} ne correspond pas au pattern {manifest.versioning.pattern}")
    return match.group("version")


def _build_tagged_tree(file_path: Path, manifest: SourceManifest, order: list[str]) -> DocumentTree:
    builder = select_builder(str(file_path))
    if builder is None:
        raise ValueError(f"Aucun builder DOM ne supporte {file_path}")
    tree = builder.build(str(file_path), manifest)
    tree.compute_all_hashes()
    tag = _extract_version_tag(file_path, manifest)
    root = tree.nodes[tree.root_id]
    root.version_tag = tag
    root.version_order = order.index(tag) if tag in order else None
    return tree


def process_manifest(manifest: SourceManifest, raw_dir: Path, engine: VersionDiffEngine) -> None:
    order = manifest.versioning.order or []
    if not order:
        print(f"  [{manifest.source_id}] ignoré : versioning.order non déclaré")
        return

    files = _resolve_files(raw_dir, manifest)
    if not files:
        print(f"  [{manifest.source_id}] ignoré : aucun fichier trouvé sous {raw_dir}")
        return

    trees: dict[str, DocumentTree] = {}
    for file_path in files:
        tree = _build_tagged_tree(file_path, manifest, order)
        tag = tree.nodes[tree.root_id].version_tag
        trees[tag] = tree
        print(f"  parsed {file_path.name} -> version_tag={tag!r} order={tree.nodes[tree.root_id].version_order}")

    missing = [t for t in order if t not in trees]
    if missing:
        print(f"  ATTENTION [{manifest.source_id}]: versions déclarées mais absentes du corpus: {missing}")

    present_ordered = [t for t in order if t in trees]
    pairs = list(combinations(present_ordered, 2))
    print(f"\n  {len(pairs)} paires à calculer pour {manifest.source_id}:\n")
    for v_from, v_to in pairs:
        report = engine.diff(trees[v_from], trees[v_to])
        by_type = {"added": 0, "removed": 0, "modified": 0}
        for change in report.changes:
            by_type[change.change_type] += 1
        print(
            f"    {v_from} -> {v_to}: {len(report.changes)} changements "
            f"(added={by_type['added']}, removed={by_type['removed']}, modified={by_type['modified']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", default="tests/fixtures/manifests")
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument(
        "--source-id",
        default=None,
        help="Ne traiter que ce source_id (sinon : tous les manifestes versionnés découverts)",
    )
    args = parser.parse_args()

    manifests = discover_manifests(Path(args.manifest_dir))
    versioned = [m for m in manifests if m.versioning.strategy != "none"]
    if args.source_id:
        versioned = [m for m in versioned if m.source_id == args.source_id]
        if not versioned:
            raise SystemExit(f"Aucun manifeste versionné trouvé pour source_id={args.source_id}")

    if not versioned:
        raise SystemExit(
            f"Aucun manifeste avec versioning.strategy != 'none' trouvé sous {args.manifest_dir}"
        )

    engine = VersionDiffEngine()
    for manifest in versioned:
        process_manifest(manifest, Path(args.raw_dir), engine)

    print(f"\nDiffs écrits dans {engine.diff_dir}/")


if __name__ == "__main__":
    main()
