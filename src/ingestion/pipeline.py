"""Orchestrateur complet d'ingestion : raw/ -> DOM -> chunks -> embeddings -> Qdrant.

Découverte de sources 100% automatique — plus de manifestes à écrire pour
ajouter un nouveau document. Tout dossier de raw/ qui contient directement
au moins un fichier supporté par un builder DOM devient une source
(source_id = nom du dossier). La seule exception déclarable reste le
versioning (voir SourceConfig / discover_sources) — un fichier
tests/fixtures/versioning/<source_id>.yaml optionnel, jamais dans raw/ qui
reste en lecture seule.
"""

import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.config.source_config import SourceConfig
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.builders.docling_builder import DoclingBuilder
from src.dom.builders.json_builder import JSONBuilder
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.ingestion.chunking.hierarchical import HierarchicalChunker
from src.ingestion.metadata import Chunk

logger = structlog.get_logger(__name__)

_BUILDERS: list[AbstractDOMBuilder] = [
    DoclingBuilder(),
    YAMLBuilder(),
    MarkdownBuilder(),
    JSONBuilder(),
]


def discover_sources(raw_dir: Path, versioning_dir: Path | None = None) -> list[SourceConfig]:
    """Découvre automatiquement les sources dans raw_dir.

    Tout dossier contenant directement au moins un fichier supporté par un
    builder DOM (peu importe la profondeur) devient une source, avec
    source_id = nom du dossier (assaini). Un override de versioning
    optionnel est chargé depuis versioning_dir/<source_id>.yaml — jamais
    depuis raw_dir lui-même (lecture seule, voir CLAUDE.md).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    source_dirs: dict[Path, str] = {}
    used_ids: set[str] = set()
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or select_builder(str(path)) is None:
            continue
        parent = path.parent
        if parent in source_dirs:
            continue
        source_id = _derive_source_id(parent, used_ids)
        used_ids.add(source_id)
        source_dirs[parent] = source_id

    sources = []
    for dir_path, source_id in sorted(source_dirs.items(), key=lambda kv: kv[1]):
        version_pattern, version_order = _load_versioning_override(source_id, versioning_dir)
        sources.append(
            SourceConfig(
                source_id=source_id,
                source_dir=dir_path,
                version_pattern=version_pattern,
                version_order=version_order,
            )
        )
        logger.info(
            "source_discovered",
            source_id=source_id,
            dir=str(dir_path),
            versioned=version_pattern is not None,
        )
    return sources


def _derive_source_id(dir_path: Path, used_ids: set[str]) -> str:
    """source_id = nom du dossier, assaini pour respecter ^[a-zA-Z0-9_]+$.
    En cas de collision (deux dossiers de même nom à des profondeurs
    différentes), élargit avec le chemin relatif complet plutôt que
    d'attribuer un simple compteur opaque."""
    base = re.sub(r"[^a-zA-Z0-9_]", "_", dir_path.name.lower()).strip("_") or "source"
    if base not in used_ids:
        return base
    widened = re.sub(r"[^a-zA-Z0-9_]", "_", "_".join(dir_path.parts[-2:]).lower()).strip("_")
    return widened if widened not in used_ids else f"{base}_{len(used_ids)}"


def _load_versioning_override(source_id: str, versioning_dir: Path | None) -> tuple[str | None, list[str] | None]:
    versioning_dir = versioning_dir or get_settings().versioning_dir
    override_path = versioning_dir / f"{source_id}.yaml"
    if not override_path.exists():
        return None, None
    data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    return data.get("pattern"), data.get("order")


class IngestionReport(BaseModel):
    """Rapport d'exécution du pipeline d'ingestion."""

    total_documents: int = 0
    total_chunks: int = 0
    errors: list[str] = Field(default_factory=list)
    by_source: dict[str, dict[str, Any]] = Field(default_factory=dict)


def run_ingestion(
    raw_dir: Path,
    store,
    embedder,
    source_id_filter: str | None = None,
    file_filter: str | None = None,
    prefix_method: str = "deterministic",
    llm_client=None,
    llm_config=None,
    versioning_dir: Path | None = None,
) -> IngestionReport:
    """Pipeline complet d'ingestion.

    Args:
        prefix_method: "deterministic" (défaut — titres ancêtres tronqués, pas
            d'appel LLM) ou "llm" (résumé du contexte parent via *llm_client*,
            avec repli automatique sur le déterministe en cas d'échec — voir
            HierarchicalChunker). Le défaut reste déterministe pour des raisons
            de reproductibilité, de vitesse d'ingestion (des milliers de chunks)
            et de sobriété mémoire GPU (voir CLAUDE.md / spec §6).
        llm_client, llm_config: Requis (tous les deux) si prefix_method="llm" ;
            ignorés sinon.
    """
    report = IngestionReport()
    sources = discover_sources(raw_dir, versioning_dir=versioning_dir)

    if source_id_filter:
        sources = [s for s in sources if s.source_id == source_id_filter]
        if not sources:
            raise ValueError(f"No source found with source_id={source_id_filter}")

    for source in sources:
        source_report = {"documents": 0, "chunks": 0, "errors": []}
        try:
            files = _resolve_files(source)
            if file_filter:
                files = [f for f in files if file_filter in f.name]

            for file_path in files:
                try:
                    settings = get_settings()
                    cache_dir = settings.cache_dir / "chunks" / source.source_id
                    cache_dir.mkdir(parents=True, exist_ok=True)

                    import hashlib
                    # Chunking universel désormais (plus de politique par
                    # source) — la clé de cache n'a plus besoin d'en tenir
                    # compte, seulement le fichier et la méthode de préfixe.
                    mtime = file_path.stat().st_mtime
                    key_str = f"{file_path}_{mtime}_{prefix_method}"
                    cache_key = hashlib.sha256(key_str.encode()).hexdigest()
                    cache_file = cache_dir / f"{file_path.name}_{cache_key}.json"

                    if cache_file.exists():
                        logger.info("loading_chunks_from_cache", file=str(file_path))
                        import json
                        with open(cache_file, "r", encoding="utf-8") as f:
                            chunks_data = json.load(f)
                            chunks = [Chunk.model_validate(c) for c in chunks_data]
                    else:
                        builder = select_builder(str(file_path))
                        if builder is None:
                            msg = f"No builder supports {file_path}"
                            source_report["errors"].append(msg)
                            continue

                        logger.info("parsing_file", file=str(file_path))
                        tree = builder.build(str(file_path), source)
                        tree.compute_all_hashes()

                        logger.info("chunking_file", file=str(file_path), prefix_method=prefix_method)
                        use_llm = prefix_method == "llm"
                        chunker = HierarchicalChunker(
                            llm_client=llm_client if use_llm else None,
                            llm_config=llm_config if use_llm else None,
                            prefix_method=prefix_method,
                            max_prefix_tokens=settings.chunk_contextual_prefix_max_tokens,
                        )
                        chunks = chunker.chunk(tree, source_type=source.source_type)

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
            report.by_source[source.source_id] = source_report

        except Exception as exc:
            msg = f"Source {source.source_id}: {exc}"
            report.errors.append(msg)
            logger.error("ingestion_source_failed", source_id=source.source_id, error=str(exc))

    return report


def _resolve_files(source: SourceConfig) -> list[Path]:
    """Tous les fichiers supportés directement dans le dossier de la source
    (pas de récursion dans les sous-dossiers — un sous-dossier qui contient
    lui-même des fichiers supportés est sa propre source, découverte
    séparément par discover_sources)."""
    if source.source_dir is None or not source.source_dir.exists():
        return []
    return sorted(p for p in source.source_dir.iterdir() if p.is_file() and select_builder(str(p)) is not None)


def select_builder(file_path: str) -> AbstractDOMBuilder | None:
    """Sélectionne le premier builder compatible."""
    for builder in _BUILDERS:
        if builder.supports(file_path):
            return builder
    return None
