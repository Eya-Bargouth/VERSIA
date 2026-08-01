"""Schémas Pydantic pour la validation des manifestes de source."""

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChunkingPolicy(BaseModel):
    """Politique de chunking déclarative."""

    semantic_unit: Literal[
        "api_endpoint", "paragraph", "list_item", "table", "document", "code_block", "heading"
    ]
    include_parent_context: bool = True
    group_by_heading: bool = False
    group_nested: list[str] = Field(default_factory=list)
    preserve_numbering: bool = False
    reformulate_natural: bool = False
    include_parameters: bool = False
    include_responses: bool = False


class VersioningConfig(BaseModel):
    """Configuration du versioning par source."""

    strategy: Literal["none", "filename_pattern", "directory_pattern", "header_pattern"] = "none"
    pattern: str | None = None
    order: list[str] | None = None


class ValidityConfig(BaseModel):
    """Validité temporelle d'une source."""

    valid_from: date | None = None
    valid_until: date | None = None
    status: Literal["active", "superseded", "deprecated", "draft"] = "active"


class SourceManifest(BaseModel):
    """Manifeste de source générique, validé au chargement."""

    manifest_version: str = "1.0.0"
    source_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_]+$")
    source_type: str = "document"  # libre, jamais un enum figé (ex: "api_spec", "regulation")
    parser: Literal["docling", "yaml_structured", "markdown", "json"]
    scope: dict[str, Any] = Field(default_factory=dict)
    chunking_policy: ChunkingPolicy = Field(default_factory=lambda: ChunkingPolicy(semantic_unit="paragraph"))
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    validity: ValidityConfig = Field(default_factory=ValidityConfig)
    rationale: str = ""
    approved_by: str = ""
    approved_date: date | None = None

    @field_validator("scope")
    @classmethod
    def _scope_must_have_include(cls, v: dict) -> dict:
        if "include" not in v:
            raise ValueError("scope must contain 'include' key with glob patterns")
        if not isinstance(v["include"], list):
            raise ValueError("scope['include'] must be a list of strings")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "SourceManifest":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
