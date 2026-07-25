"""HierarchicalChunker — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

"""Chunking hiérarchique avec Contextual Retrieval (déterministe + LLM)."""

import hashlib
from uuid import uuid4

import structlog

from src.config.manifest_schema import ChunkingPolicy
from src.dom.models import DOMNode, DocumentTree, NodeType
from src.dom.utils import build_hierarchy_path, get_structural_ancestors
from src.ingestion.chunking.base import ChunkingStrategy
from src.ingestion.metadata import Chunk, ChunkMetadata
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)

# Types de nœuds considérés comme unités sémantiques feuilles
_CONTENT_NODE_TYPES = {
    NodeType.API_ENDPOINT,
    NodeType.PARAGRAPH,
    NodeType.LIST_ITEM,
    NodeType.TABLE,
    NodeType.CODE_BLOCK,
    NodeType.DOCUMENT,
}


class HierarchicalChunker(ChunkingStrategy):
    """Chunker hiérarchique produisant des chunks parents/enfants/feuilles.

    Chaque chunk feuille porte un contextual_prefix résumant son contexte parent.
    """

    def __init__(
            self,
            llm_client: BaseLLMClient | None = None,
            llm_config: LLMConfig | None = None,
            prefix_method: str = "deterministic",
            max_prefix_tokens: int = 100,
    ):
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.prefix_method = prefix_method
        self.max_prefix_tokens = max_prefix_tokens

    def chunk(self, tree: DocumentTree, policy: ChunkingPolicy) -> list[Chunk]:
        """Orchestre le chunking hiérarchique complet."""
        chunks: list[Chunk] = []
        chunk_map: dict[str, Chunk] = {}  # mapping node_id -> chunk pour liens parent/enfant

        # 1. Identifier les nœuds feuilles (unités sémantiques)
        leaf_nodes = self._collect_leaf_nodes(tree, policy)
        logger.info("chunking_leaves_identified", count=len(leaf_nodes))

        # 2. Créer un chunk par nœud feuille
        for node in leaf_nodes:
            chunk = self._create_leaf_chunk(tree, node, policy)
            chunks.append(chunk)
            chunk_map[str(node.id)] = chunk

        # 3. Créer les chunks parents structurels (SECTION, API_SCHEMA, etc.)
        structural_chunks = self._create_structural_chunks(tree, leaf_nodes, chunk_map, policy)
        chunks.extend(structural_chunks)

        # 4. Lier la hiérarchie parent/enfant entre chunks
        self._link_chunk_hierarchy(chunks, chunk_map, tree)

        # 5. Calculer les content_hash
        for chunk in chunks:
            chunk.content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

        logger.info("chunking_complete", total_chunks=len(chunks), leaves=len(leaf_nodes))
        return chunks

    def _collect_leaf_nodes(self, tree: DocumentTree, policy: ChunkingPolicy) -> list[DOMNode]:
        """Retourne les nœuds à transformer en chunks feuilles selon la politique."""
        candidates = []
        for node in tree.nodes.values():
            if not node.is_content:
                continue
            # Filtrage par semantic_unit
            if policy.semantic_unit == "api_endpoint" and node.type == NodeType.API_ENDPOINT:
                candidates.append(node)
            elif policy.semantic_unit == "paragraph" and node.type == NodeType.PARAGRAPH:
                candidates.append(node)
            elif policy.semantic_unit == "list_item" and node.type == NodeType.LIST_ITEM:
                candidates.append(node)
            elif policy.semantic_unit == "table" and node.type == NodeType.TABLE:
                candidates.append(node)
            elif policy.semantic_unit == "code_block" and node.type == NodeType.CODE_BLOCK:
                candidates.append(node)
            elif policy.semantic_unit == "document" and node.type == NodeType.DOCUMENT:
                candidates.append(node)
            elif policy.semantic_unit == "heading" and node.type == NodeType.HEADING:
                candidates.append(node)
        # Ordonner par ordre d'apparition dans l'arbre (DFS approximatif)
        candidates.sort(key=lambda n: (n.page_num or 0, n.line_num or 0, str(n.id)))
        return candidates

    def _create_leaf_chunk(self, tree: DocumentTree, node: DOMNode, policy: ChunkingPolicy) -> Chunk:
        """Crée un chunk feuille avec contextual_prefix."""
        raw_text = node.markdown or node.text or ""
        hierarchy_path = build_hierarchy_path(tree, node.id)

        # Contextual Retrieval : générer le préfixe
        if policy.include_parent_context:
            prefix = self._generate_contextual_prefix(tree, node, policy)
        else:
            prefix = ""

        text = f"{prefix}\n\n{raw_text}" if prefix else raw_text

        metadata = ChunkMetadata(
            source_type="unknown",  # sera surchargé par le pipeline si besoin
            source_id=node.source_id,
            node_type=node.type,
            hierarchy_path=hierarchy_path,
            format_original=tree.source_path.split(".")[-1],
            section_title=next(
                (a.metadata.get("title") or a.text for a in get_structural_ancestors(tree, node.id) if
                 a.type == NodeType.HEADING),
                None,
            ),
            page_num=node.page_num,
            line_num=node.line_num,
            char_offset=node.char_offset,
        )

        return Chunk(
            source_id=node.source_id,
            node_ids=[str(node.id)],
            text=text,
            raw_text=raw_text,
            contextual_prefix=prefix,
            level=node.level,
            metadata=metadata,
            hierarchy_path=hierarchy_path,
            version_tag=node.version_tag,
            version_order=node.version_order,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            status=node.status,
        )

    def _generate_contextual_prefix(self, tree: DocumentTree, node: DOMNode, policy: ChunkingPolicy) -> str:
        """Génère le préfixe contextuel (déterministe par défaut, LLM en option)."""
        if self.prefix_method == "llm" and self.llm_client is not None and self.llm_config is not None:
            return self._generate_prefix_with_llm(tree, node)
        # Méthode déterministe par défaut (titres des ancêtres structurels)
        ancestors = get_structural_ancestors(tree, node.id)
        titles = []
        for a in ancestors:
            title = a.metadata.get("title") or a.text or a.markdown or a.type.value
            if title:
                titles.append(title.strip()[:60])
        prefix = " > ".join(titles)
        # Tronquer approximativement à max_prefix_tokens (estimation 4 chars/token)
        max_chars = self.max_prefix_tokens * 4
        if len(prefix) > max_chars:
            prefix = prefix[:max_chars] + "..."
        return prefix

    def _generate_prefix_with_llm(self, tree: DocumentTree, node: DOMNode) -> str:
        """Appelle le LLM local pour résumer le contexte du chunk."""
        ancestors = get_structural_ancestors(tree, node.id)
        context = "\n".join(
            f"- {a.metadata.get('title') or a.text or a.type.value}" for a in ancestors
        )
        prompt = (
            f"Résume en une phrase le contexte documentaire suivant (max {self.max_prefix_tokens} tokens):\n"
            f"{context}\n\nRésumé:"
        )
        try:
            resp = self.llm_client.complete(
                [LLMMessage(role="user", content=prompt)],
                self.llm_config,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning("llm_prefix_failed", error=str(exc), fallback="deterministic")
            return self._generate_contextual_prefix(tree, node, ChunkingPolicy(include_parent_context=True))

    def _create_structural_chunks(
            self,
            tree: DocumentTree,
            leaf_nodes: list[DOMNode],
            chunk_map: dict[str, Chunk],
            policy: ChunkingPolicy,
    ) -> list[Chunk]:
        """Crée des chunks parents pour les nœuds structurels (SECTION, API_SCHEMA, etc.)."""
        structural_ids = {str(n.id) for n in leaf_nodes}
        structural_chunks = []
        for node in tree.nodes.values():
            if not node.is_structural:
                continue
            if str(node.id) in structural_ids:
                continue
            # Ne créer un chunk structural que s'il a des descendants feuilles
            has_leaf_descendant = any(
                str(desc.id) in structural_ids for desc in _get_descendants(tree, node.id)
            )
            if not has_leaf_descendant:
                continue
            hierarchy_path = build_hierarchy_path(tree, node.id)
            raw_text = node.markdown or node.text or ""
            metadata = ChunkMetadata(
                source_type="unknown",
                source_id=node.source_id,
                node_type=node.type,
                hierarchy_path=hierarchy_path,
                format_original=tree.source_path.split(".")[-1],
                section_title=node.metadata.get("title") or node.text,
                page_num=node.page_num,
            )
            chunk = Chunk(
                source_id=node.source_id,
                node_ids=[str(node.id)],
                text=raw_text,
                raw_text=raw_text,
                contextual_prefix="",
                level=node.level,
                metadata=metadata,
                hierarchy_path=hierarchy_path,
                version_tag=node.version_tag,
                version_order=node.version_order,
                valid_from=node.valid_from,
                valid_until=node.valid_until,
                status=node.status,
            )
            structural_chunks.append(chunk)
            chunk_map[str(node.id)] = chunk
        return structural_chunks

    def _link_chunk_hierarchy(self, chunks: list[Chunk], chunk_map: dict[str, Chunk], tree: DocumentTree) -> None:
        """Établit les liens parent_chunk_id / child_chunk_ids entre chunks."""
        # Inverser le mapping : chunk_id -> node_id
        chunk_to_node = {c.chunk_id: c.node_ids[0] for c in chunks if c.node_ids}
        node_to_chunk = {nid: c for nid, c in chunk_map.items()}

        for chunk in chunks:
            if not chunk.node_ids:
                continue
            node_id = chunk.node_ids[0]
            node = tree.nodes.get(node_id)
            if node is None:
                continue
            # Parent
            if node.parent_id is not None:
                parent_chunk = node_to_chunk.get(str(node.parent_id))
                if parent_chunk is not None:
                    chunk.parent_chunk_id = parent_chunk.chunk_id
                    if chunk.chunk_id not in parent_chunk.child_chunk_ids:
                        parent_chunk.child_chunk_ids.append(chunk.chunk_id)


def _get_descendants(tree: DocumentTree, node_id) -> list[DOMNode]:
    """Parcours DFS des descendants."""
    from src.dom.utils import get_descendants
    return get_descendants(tree, node_id)