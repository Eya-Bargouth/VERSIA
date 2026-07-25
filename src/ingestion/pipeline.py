"""Orchestrateur complet d'ingestion : manifestes -> DOM -> chunks -> embeddings -> Qdrant."""

from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.config.manifest_schema import SourceManifest
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.builders.docling_builder import DoclingBuilder
from src.dom.builders.json_builder import JSONBuilder
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.ingestion.chunking.policies import resolve_policy

logger = structlog.get_logger(__name__)

_BUILDERS: list[AbstractDOMBuilder] = [
    DoclingBuilder(),
    YAMLBuilder(),
    MarkdownBuilder(),
    JSONBuilder(),
]


def discover_manifests(manifest_dir: Path) -> list[SourceManifest]:
    """Parcourt manifest_dir, charge et valide chaque fichier .yaml/.yml."""
    manifests = []
    errors = []

    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    for path in list(manifest_dir.glob("*.yaml")) + list(manifest_dir.glob("*.yml")):
        try:
            manifest = SourceManifest.from_yaml(path)
            manifests.append(manifest)
            logger.info("manifest_loaded", source_id=manifest.source_id, path=str(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            logger.error("manifest_invalid", path=str(path), error=str(exc))

    if errors:
        logger.warning("manifest_discovery_errors", count=len(errors), errors=errors)

    return manifests


class IngestionReport(BaseModel):
    """Rapport d'exécution du pipeline d'ingestion."""

    total_documents: int = 0
    total_chunks: int = 0
    errors: list[str] = Field(default_factory=list)
    by_source: dict[str, dict[str, Any]] = Field(default_factory=dict)


def run_ingestion(
    manifest_dir: Path,
    raw_dir: Path,
    store,
    embedder,
    source_id_filter: str | None = None,
) -> IngestionReport:
    """Pipeline complet d'ingestion."""
    report = IngestionReport()
    manifests = discover_manifests(manifest_dir)

    if source_id_filter:
        manifests = [m for m in manifests if m.source_id == source_id_filter]
        if not manifests:
            raise ValueError(f"No manifest found with source_id={source_id_filter}")

    for manifest in manifests:
        source_report = {"documents": 0, "chunks": 0, "errors": []}
        try:
            files = _resolve_files(raw_dir, manifest)
            for file_path in files:
                try:
                    builder = _select_builder(str(file_path))
                    if builder is None:
                        msg = f"No builder supports {file_path}"
                        source_report["errors"].append(msg)
                        continue

                    tree = builder.build(str(file_path), manifest)
                    tree.compute_all_hashes()

                    chunker = HierarchicalChunker(
                        prefix_method="deterministic",
                        max_prefix_tokens=100,
                    )
                    policy = resolve_policy(manifest.chunking_policy)
                    chunks = chunker.chunk(tree, policy)

                    embeddings = embedder.embed([c.text for c in chunks])

                    store.upsert(chunks, embeddings)

                    source_report["documents"] += 1
                    source_report["chunks"] += len(chunks)
                    report.total_chunks += len(chunks)

                except Exception as exc:
                    msg = f"{file_path}: {exc}"
                    source_report["errors"].append(msg)
                    logger.error("ingestion_file_failed", path=str(file_path), error=str(exc))

            report.total_documents += source_report["documents"]
            report.by_source[manifest.source_id] = source_report

        except Exception as exc:
            msg = f"Manifest {manifest.source_id}: {exc}"
            report.errors.append(msg)
            logger.error("ingestion_manifest_failed", source_id=manifest.source_id, error=str(exc))

    return report


def _resolve_files(raw_dir: Path, manifest: SourceManifest) -> list[Path]:
    """Résout les fichiers correspondant au scope d'un manifeste."""
    source_dir = raw_dir / manifest.source_id
    if not source_dir.exists():
        source_dir = raw_dir

    include_patterns = manifest.scope.get("include", [])
    exclude_patterns = manifest.scope.get("exclude", [])

    files = set()
    for pattern in include_patterns:
        files.update(source_dir.glob(pattern))
        if not files and not source_dir.exists():
            files.update(raw_dir.glob(pattern))

    for pattern in exclude_patterns:
        files -= set(source_dir.glob(pattern))

    return sorted(files)


def _select_builder(file_path: str) -> AbstractDOMBuilder | None:
    """Sélectionne le premier builder compatible."""
    for builder in _BUILDERS:
        if builder.supports(file_path):
            return builder
    return None