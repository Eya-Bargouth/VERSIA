"""Document Object Model unifié pour TRADE."""

from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    # Only for static type checking — importing these at runtime would create
    # a circular import (src.ingestion.chunking.* already imports from
    # src.dom.models). See DocumentTree.to_chunks() below.
    from src.config.manifest_schema import ChunkingPolicy
    from src.ingestion.chunking.base import ChunkingStrategy
    from src.ingestion.metadata import Chunk


class NodeType(StrEnum):
    """Types de nœuds DOM — structure documentaire uniquement."""

    # Structurels
    DOCUMENT = "document"
    PAGE = "page"
    SECTION = "section"
    SUBSECTION = "subsection"
    HEADING = "heading"

    # Contenu
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"

    # Media
    FIGURE = "figure"
    IMAGE = "image"
    CAPTION = "caption"

    # Spécifiques formats structurés (syntaxe, pas sémantique)
    API_ENDPOINT = "api_endpoint"
    API_PARAMETER = "api_parameter"
    API_RESPONSE = "api_response"
    API_SCHEMA = "api_schema"

    # Métadonnées
    METADATA = "metadata"
    FOOTNOTE = "footnote"


class BoundingBox(BaseModel):
    """Position physique dans le document source."""

    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    page: int | None = None


class TextStyle(BaseModel):
    """Style typographique."""

    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_italic: bool = False
    color: str | None = None


class DOMNode(BaseModel):
    """Nœud unique dans l'arbre DOM."""

    id: UUID = Field(default_factory=uuid4)
    type: NodeType
    source_id: str
    source_path: str

    text: str | None = None
    markdown: str | None = None
    html: str | None = None

    parent_id: UUID | None = None
    children_ids: list[UUID] = Field(default_factory=list)
    level: int = 0

    bbox: BoundingBox | None = None
    style: TextStyle | None = None

    metadata: dict = Field(default_factory=dict)

    page_num: int | None = None
    line_num: int | None = None
    char_offset: int | None = None

    version_tag: str | None = None
    version_order: int | None = None

    valid_from: date | None = None
    valid_until: date | None = None
    status: Literal["active", "superseded", "deprecated", "draft"] | None = None

    content_hash: str | None = None

    is_structural: bool = False
    is_content: bool = True

    @field_validator("children_ids")
    @classmethod
    def _no_self_reference(cls, v: list[UUID], info) -> list[UUID]:
        data = info.data
        if "id" in data and data["id"] in v:
            raise ValueError("A node cannot be its own child")
        return v

    def compute_hash(self) -> str:
        """Calcule le SHA-256 du contenu textuel (markdown prioritaire, puis text, puis html)."""
        content = self.markdown or self.text or self.html or ""
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.content_hash


class DocumentTree(BaseModel):
    """Arbre DOM complet avec index plat O(1)."""

    root_id: UUID
    nodes: dict[UUID, DOMNode] = Field(default_factory=dict)
    source_id: str
    source_path: str

    def get_node(self, node_id: UUID) -> DOMNode | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: UUID) -> list[DOMNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_parent(self, node_id: UUID) -> DOMNode | None:
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.nodes.get(node.parent_id)

    def get_path(self, node_id: UUID) -> list[DOMNode]:
        """Chemin depuis la racine jusqu'au nœud (inclus)."""
        path = []
        current = self.nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self.get_parent(current.id)
        path.reverse()
        return path

    def get_siblings(self, node_id: UUID) -> list[DOMNode]:
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        parent = self.nodes.get(node.parent_id)
        if parent is None:
            return []
        return [
            self.nodes[cid]
            for cid in parent.children_ids
            if cid != node_id and cid in self.nodes
        ]

    def get_leaves(self) -> list[DOMNode]:
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_nodes_by_type(self, node_type: NodeType) -> list[DOMNode]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def add_node(self, node: DOMNode, parent_id: UUID | None = None) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        # Respect existing node.parent_id if no explicit parent_id argument is given
        effective_parent_id = parent_id if parent_id is not None else node.parent_id
        node.parent_id = effective_parent_id
        if effective_parent_id is not None and effective_parent_id in self.nodes:
            parent = self.nodes[effective_parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        self.nodes[node.id] = node

    def compute_all_hashes(self) -> None:
        for node in self.nodes.values():
            node.compute_hash()

    def to_chunks(
        self, strategy: "ChunkingStrategy", policy: "ChunkingPolicy", source_type: str = "unknown"
    ) -> list["Chunk"]:
        """Délègue le chunking à *strategy* (contrat spec §12.1/§12.2).

        Thin wrapper kept here for API-contract parity with the spec — the
        actual chunking logic lives in src/ingestion/chunking/ (e.g.
        HierarchicalChunker) so that src/dom/ stays decoupled from the
        ingestion layer (no runtime import of ChunkingStrategy here; see the
        TYPE_CHECKING guard above). Equivalent to
        ``strategy.chunk(tree, policy, source_type=source_type)``.
        """
        return strategy.chunk(self, policy, source_type=source_type)

    def validate_integrity(self) -> list[str]:
        """Retourne une liste d'erreurs si l'arbre est corrompu."""
        errors = []
        if self.root_id not in self.nodes:
            errors.append("Root node missing from index")
        for nid, node in self.nodes.items():
            if node.parent_id is not None and node.parent_id not in self.nodes:
                errors.append(f"Node {nid} references missing parent {node.parent_id}")
            for cid in node.children_ids:
                if cid not in self.nodes:
                    errors.append(f"Node {nid} references missing child {cid}")
        return errors