#!/usr/bin/env python3
"""Calcule les diffs entre toutes les paires de versions de chaque source
versionnée découverte automatiquement sous raw_dir, et les persiste dans
data/diffs/ via VersionDiffEngine.

Générique par construction : découvre toutes les sources dont un override de
versioning existe (voir tests/fixtures/versioning/<source_id>.yaml — jamais
dans raw/, en lecture seule), peu importe le format (yaml/json/markdown/pdf)
ou le source_id. Ajouter une nouvelle source versionnée ne demande aucune
modification de ce script — seulement un fichier
tests/fixtures/versioning/<source_id>.yaml déclarant `pattern`/`order`.

Les builders DOM peuplent DOMNode.version_tag/version_order depuis
SourceConfig.version_pattern au moment du build() (AbstractDOMBuilder.
_create_root_node) : ce script se contente de construire l'arbre et de lire
le tag qui en résulte, plus besoin de le dériver lui-même.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.source_config import SourceConfig
from src.dom.models import DocumentTree
from src.ingestion.pipeline import discover_sources, select_builder
from src.ingestion.version_diff import VersionDiffEngine


def _build_tagged_tree(file_path: Path, source: SourceConfig) -> DocumentTree:
    builder = select_builder(str(file_path))
    if builder is None:
        raise ValueError(f"Aucun builder DOM ne supporte {file_path}")
    tree = builder.build(str(file_path), source)
    tree.compute_all_hashes()
    if tree.nodes[tree.root_id].version_tag is None:
        raise ValueError(
            f"{file_path.name} : version_tag non résolu — vérifier le pattern de "
            f"tests/fixtures/versioning/{source.source_id}.yaml"
        )
    return tree


def process_source(source: SourceConfig, engine: VersionDiffEngine) -> None:
    order = source.version_order or []
    if not order:
        print(f"  [{source.source_id}] ignoré : aucun 'order' déclaré dans l'override de versioning")
        return

    files = sorted(p for p in (source.source_dir or Path()).iterdir() if p.is_file()) if source.source_dir else []
    if not files:
        print(f"  [{source.source_id}] ignoré : aucun fichier trouvé sous {source.source_dir}")
        return

    trees: dict[str, DocumentTree] = {}
    for file_path in files:
        tree = _build_tagged_tree(file_path, source)
        tag = tree.nodes[tree.root_id].version_tag
        trees[tag] = tree
        print(f"  parsed {file_path.name} -> version_tag={tag!r} order={tree.nodes[tree.root_id].version_order}")

    missing = [t for t in order if t not in trees]
    if missing:
        print(f"  ATTENTION [{source.source_id}]: versions déclarées mais absentes du corpus: {missing}")

    present_ordered = [t for t in order if t in trees]
    pairs = list(combinations(present_ordered, 2))
    print(f"\n  {len(pairs)} paires à calculer pour {source.source_id}:\n")
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
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--versioning-dir", default="tests/fixtures/versioning")
    parser.add_argument(
        "--source-id",
        default=None,
        help="Ne traiter que ce source_id (sinon : toutes les sources versionnées découvertes)",
    )
    args = parser.parse_args()

    sources = discover_sources(Path(args.raw_dir), versioning_dir=Path(args.versioning_dir))
    versioned = [s for s in sources if s.version_pattern is not None]
    if args.source_id:
        versioned = [s for s in versioned if s.source_id == args.source_id]
        if not versioned:
            raise SystemExit(f"Aucune source versionnée trouvée pour source_id={args.source_id}")

    if not versioned:
        raise SystemExit(f"Aucune source versionnée trouvée sous {args.versioning_dir}")

    engine = VersionDiffEngine()
    for source in versioned:
        process_source(source, engine)

    print(f"\nDiffs écrits dans {engine.diff_dir}/")


if __name__ == "__main__":
    main()
