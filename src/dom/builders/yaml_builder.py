"""Builder YAML — parcours générique (voir StructuredDataBuilder, partagée
avec JSONBuilder) — ce module ne fait que choisir le chargeur YAML.
"""

from pathlib import Path
from typing import Any

import yaml

from src.dom.builders.structured_data import StructuredDataBuilder


class YAMLBuilder(StructuredDataBuilder):
    """Builder pour fichiers .yaml/.yml, de forme quelconque."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".yaml", ".yml"}

    def _load(self, source_path: str) -> Any:
        with open(source_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
