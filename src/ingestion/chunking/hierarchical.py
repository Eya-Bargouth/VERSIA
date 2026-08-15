"""HierarchicalChunker — Phase 1 (placeholder)."""

# TODO: Phase 2 - implementer

"""Chunking hiérarchique avec Contextual Retrieval (déterministe + LLM)."""

import hashlib
from uuid import UUID, uuid4

import structlog

from src.dom.models import DOMNode, DocumentTree, NodeType
from src.dom.utils import build_hierarchy_path, get_structural_ancestors
from src.ingestion.chunking.base import ChunkingStrategy
from src.ingestion.metadata import Chunk, ChunkMetadata
from src.llm.interface import BaseLLMClient, LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)


class HierarchicalChunker(ChunkingStrategy):
    """Chunker hiérarchique produisant des chunks parents/enfants/feuilles.

    Chunking universel : tout nœud de contenu (is_content=True) devient un
    chunk feuille, quel que soit son type — plus de politique déclarative
    par source (ancien ChunkingPolicy.semantic_unit, supprimé). Chaque chunk
    feuille porte un contextual_prefix résumant son contexte parent
    (toujours actif — les 5 sources réelles du projet l'activaient déjà
    uniformément, ce n'était jamais un vrai choix par source).
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

    def chunk(self, tree: DocumentTree, source_type: str = "unknown") -> list[Chunk]:
        """Orchestre le chunking hiérarchique complet."""
        chunks: list[Chunk] = []
        chunk_map: dict[str, Chunk] = {}  # mapping node_id -> chunk pour liens parent/enfant

        # 1. Identifier les nœuds feuilles (tout nœud de contenu)
        leaf_nodes = self._collect_leaf_nodes(tree)
        logger.info("chunking_leaves_identified", count=len(leaf_nodes))

        # 2. Créer un chunk par nœud feuille
        for node in leaf_nodes:
            chunk = self._create_leaf_chunk(tree, node, source_type)
            chunks.append(chunk)
            chunk_map[str(node.id)] = chunk

        # 3. Créer les chunks parents structurels (SECTION, API_SCHEMA, etc.)
        structural_chunks = self._create_structural_chunks(tree, leaf_nodes, chunk_map, source_type)
        chunks.extend(structural_chunks)

        # 4. Lier la hiérarchie parent/enfant entre chunks
        self._link_chunk_hierarchy(chunks, chunk_map, tree)

        # 5. Calculer les content_hash — inclut version_tag pour éviter que deux
        # chunks au texte identique mais de versions différentes (le cas
        # courant : un endpoint Stripe inchangé entre deux versions) ne
        # collisent sur le même point Qdrant (uuid5(content_hash), voir
        # vector_store.py::upsert) et s'écrasent silencieusement l'un l'autre.
        for chunk in chunks:
            hash_input = chunk.text if chunk.version_tag is None else f"{chunk.text}\x00{chunk.version_tag}"
            chunk.content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        logger.info("chunking_complete", total_chunks=len(chunks), leaves=len(leaf_nodes))
        return chunks

    def _collect_leaf_nodes(self, tree: DocumentTree) -> list[DOMNode]:
        """Retourne tous les nœuds de contenu (is_content=True) — chunking
        universel, aucun type de nœud privilégié par rapport à un autre."""
        candidates = [node for node in tree.nodes.values() if node.is_content]
        # Ordonner par ordre d'apparition dans l'arbre (DFS approximatif)
        candidates.sort(key=lambda n: (n.page_num or 0, n.line_num or 0, str(n.id)))
        return candidates

    def _create_leaf_chunk(self, tree: DocumentTree, node: DOMNode, source_type: str) -> Chunk:
        """Crée un chunk feuille avec contextual_prefix."""
        root = tree.nodes[tree.root_id]
        raw_text = node.markdown or node.text or ""
        hierarchy_path = build_hierarchy_path(tree, node.id)

        # Contextual Retrieval : toujours actif (les 5 sources réelles du
        # projet l'activaient déjà uniformément — jamais un vrai choix par
        # source, voir ChunkingPolicy.include_parent_context supprimé).
        prefix = self._generate_contextual_prefix(tree, node)

        text = f"{prefix}\n\n{raw_text}" if prefix else raw_text

        metadata = ChunkMetadata(
            source_type=source_type,
            source_id=node.source_id,
            node_type=node.type,
            hierarchy_path=hierarchy_path,
            format_original=tree.source_path.split(".")[-1],
            source_path=tree.source_path,
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
            # Hérité de la racine de l'arbre : un fichier source représente
            # une seule version / une seule validité, déclarée une fois dans
            # le manifeste — jamais différente d'un nœud à l'autre du même
            # fichier (voir AbstractDOMBuilder._create_root_node).
            version_tag=node.version_tag or root.version_tag,
            version_order=node.version_order if node.version_order is not None else root.version_order,
            valid_from=node.valid_from or root.valid_from,
            valid_until=node.valid_until or root.valid_until,
            status=node.status or root.status,
        )

    def _generate_contextual_prefix(self, tree: DocumentTree, node: DOMNode) -> str:
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
            return self._generate_contextual_prefix(tree, node)

    def _create_structural_chunks(
            self,
            tree: DocumentTree,
            leaf_nodes: list[DOMNode],
            chunk_map: dict[str, Chunk],
            source_type: str,
    ) -> list[Chunk]:
        """Crée des chunks parents pour les nœuds structurels (SECTION, API_SCHEMA, etc.)."""
        structural_ids = {str(n.id) for n in leaf_nodes}
        structural_chunks = []
        root = tree.nodes[tree.root_id]
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
                source_type=source_type,
                source_id=node.source_id,
                node_type=node.type,
                hierarchy_path=hierarchy_path,
                format_original=tree.source_path.split(".")[-1],
                source_path=tree.source_path,
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
                version_tag=node.version_tag or root.version_tag,
                version_order=node.version_order if node.version_order is not None else root.version_order,
                valid_from=node.valid_from or root.valid_from,
                valid_until=node.valid_until or root.valid_until,
                status=node.status or root.status,
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
            # chunk.node_ids stocke des UUID sérialisés en str (voir
            # node_ids=[str(node.id)] à la création du chunk) ; tree.nodes
            # est indexé par UUID, pas par str — sans cette conversion,
            # .get() ne trouve jamais rien et AUCUN chunk n'obtient de
            # parent_chunk_id, silencieusement (bug réel trouvé en testant
            # l'expansion parent explicite, audit #15).
            node = tree.nodes.get(UUID(chunk.node_ids[0]))
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