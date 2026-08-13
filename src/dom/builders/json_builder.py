"""Builder JSON — parcours générique (voir StructuredDataBuilder, partagée
avec YAMLBuilder) — ce module ne fait que choisir le chargeur JSON.
"""

import json
from pathlib import Path
from typing import Any

from src.dom.builders.structured_data import StructuredDataBuilder


class JSONBuilder(StructuredDataBuilder):
    """Builder pour fichiers .json, de forme quelconque."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".json"

    def _load(self, source_path: str) -> Any:
        with open(source_path, "r", encoding="utf-8") as f:
            return json.load(f)
