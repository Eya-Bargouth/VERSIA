"""Orchestrateur complet d'ingestion : manifestes -> DOM -> chunks -> embeddings -> Qdrant."""

from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.config.manifest_schema import SourceManifest
from src.config.settings import get_settings
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.builders.docling_builder import DoclingBuilder
from src.dom.builders.json_builder import JSONBuilder
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.ingestion.chunking.policies import resolve_policy
from src.ingestion.metadata import Chunk

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
    file_filter: str | None = None,
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
            if file_filter:
                files = [f for f in files if file_filter in f.name]

            for file_path in files:
                try:
                    settings = get_settings()
                    cache_dir = settings.cache_dir / "chunks" / manifest.source_id
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    import hashlib
                    # Calculate cache key including file mtime and chunking policy
                    mtime = file_path.stat().st_mtime
                    policy_json = manifest.chunking_policy.model_dump_json()
                    key_str = f"{file_path}_{mtime}_{policy_json}"
                    cache_key = hashlib.sha256(key_str.encode()).hexdigest()
                    cache_file = cache_dir / f"{file_path.name}_{cache_key}.json"

                    if cache_file.exists():
                        logger.info("loading_chunks_from_cache", file=str(file_path))
                        import json
                        with open(cache_file, "r", encoding="utf-8") as f:
                            chunks_data = json.load(f)
                            chunks = [Chunk.model_validate(c) for c in chunks_data]
                    else:
                        builder = _select_builder(str(file_path))
                        if builder is None:
                            msg = f"No builder supports {file_path}"
                            source_report["errors"].append(msg)
                            continue

                        logger.info("parsing_file", file=str(file_path))
                        tree = builder.build(str(file_path), manifest)
                        tree.compute_all_hashes()

                        logger.info("chunking_file", file=str(file_path))
                        chunker = HierarchicalChunker(
                            prefix_method="deterministic",
                            max_prefix_tokens=100,
                        )
                        policy = resolve_policy(manifest.chunking_policy)
                        chunks = chunker.chunk(tree, policy, source_type=manifest.source_type)

                        # Save chunks to cache
                        with open(cache_file, "w", encoding="utf-8") as f:
                            import json
                            json.dump([c.model_dump(mode="json") for c in chunks], f, indent=2)

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
    """Résout les fichiers correspondant au scope d'un manifeste.

    Priorité de résolution :
    1. ``scope.base_dir`` déclaré dans le manifeste (chemin relatif à raw_dir).
    2. ``raw_dir / manifest.source_id`` si le répertoire existe.
    3. Recherche récursive (rglob) depuis raw_dir en dernier recours.
    """
    # 1. base_dir déclaratif dans le manifeste (clé optionnelle de scope)
    base_dir_relative = manifest.scope.get("base_dir")
    if base_dir_relative:
        source_dir = raw_dir / base_dir_relative
    else:
        # 2. Répertoire nommé d'après le source_id
        candidate = raw_dir / manifest.source_id
        source_dir = candidate if candidate.exists() else None

    include_patterns = manifest.scope.get("include", [])
    exclude_patterns = manifest.scope.get("exclude", [])

    files: set[Path] = set()
    for pattern in include_patterns:
        if source_dir is not None and source_dir.exists():
            matched = list(source_dir.glob(pattern))
        else:
            matched = []
        if not matched:
            # 3. Recherche récursive depuis raw_dir (dernier recours)
            matched = list(raw_dir.rglob(pattern))
        files.update(matched)

    # Exclusions : chercher dans source_dir et globalement
    for pattern in exclude_patterns:
        if source_dir is not None and source_dir.exists():
            files -= set(source_dir.glob(pattern))
        files -= set(raw_dir.rglob(pattern))

    return sorted(files)


def _select_builder(file_path: str) -> AbstractDOMBuilder | None:
    """Sélectionne le premier builder compatible."""
    for builder in _BUILDERS:
        if builder.supports(file_path):
            return builder
    return None