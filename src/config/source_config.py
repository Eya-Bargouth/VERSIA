"""Configuration minimale d'une source — remplace l'ancien SourceManifest.

Une source = un dossier de raw/ contenant directement des fichiers (peu
importe la profondeur). Tout est auto-détecté (source_id = nom du dossier,
scope = tout le contenu du dossier, chunking = universel — voir
HierarchicalChunker) — aucune déclaration manuelle nécessaire dans le cas
général.

La SEULE exception gardée : le pattern de version pour les sources
versionnées (aujourd'hui, seul Stripe). Cette information ne peut pas être
déduite sans risque réel de mal ordonner les versions (ex. tri alphabétique
qui classerait "v10" avant "v9" — voir discussion de session) — contrairement
à tout le reste du manifeste, ce n'est pas une simplification sûre à
automatiser. Cette déclaration vit dans tests/fixtures/versioning/ (jamais
dans raw/, en lecture seule), un fichier par source
versionnée, nommé <source_id>.yaml.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SourceConfig(BaseModel):
    """Configuration effective d'une source, après auto-détection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_]+$")
    # Libre, jamais un enum figé (spec/invariant projet) — générique par
    # défaut, aucune tentative de deviner la sémantique d'un corpus depuis
    # son nom de dossier.
    source_type: str = "document"
    # Dossier de raw/ dont proviennent les fichiers de cette source (utilisé
    # par le pipeline d'ingestion pour résoudre les fichiers — ignoré par
    # les builders DOM eux-mêmes, qui reçoivent directement un chemin de
    # fichier). Optionnel pour rester utilisable dans les tests qui
    # construisent un SourceConfig sans passer par discover_sources().
    source_dir: Path | None = None
    # Présent seulement si un override de versioning existe pour cette
    # source (voir tests/fixtures/versioning/<source_id>.yaml).
    version_pattern: str | None = None
    version_order: list[str] | None = None
