"""Helpers de navigation et manipulation de l'arbre DOM."""

import re

from src.dom.models import DOMNode, DocumentTree, NodeType

_TRAILING_INDEX_RE = re.compile(r"\[\d+\]$")


def get_ancestors(tree: DocumentTree, node_id) -> list[DOMNode]:
    """Retourne les ancêtres depuis le parent jusqu'à la racine."""
    ancestors = []
    current = tree.get_parent(node_id)
    while current is not None:
        ancestors.append(current)
        current = tree.get_parent(current.id)
    ancestors.reverse()
    return ancestors


def get_descendants(tree: DocumentTree, node_id) -> list[DOMNode]:
    """Parcours DFS de tous les descendants."""
    descendants = []
    stack = [node_id]
    while stack:
        current_id = stack.pop()
        children = tree.get_children(current_id)
        for child in children:
            descendants.append(child)
            stack.append(child.id)
    return descendants


def get_leaves(tree: DocumentTree) -> list[DOMNode]:
    return tree.get_leaves()


def get_nodes_by_type(tree: DocumentTree, node_type: NodeType) -> list[DOMNode]:
    return tree.get_nodes_by_type(node_type)


def build_hierarchy_path(tree: DocumentTree, node_id) -> str:
    """Construit le fil d'Ariane : 'Document > Section 2 > Subsection 2.1'.

    Non tronqué : cette valeur sert de clé d'identité (`derive_parent_path`,
    dédup par version dans `HybridRetriever._dedupe_version_siblings`) — une
    troncature par segment coupait silencieusement le suffixe `[N]` des
    chemins longs, rendant des éléments de liste distincts indiscernables
    entre eux (paramètres d'un même endpoint fusionnés à tort). Un usage
    d'affichage tronqué existe séparément (voir
    `HierarchicalChunker._generate_contextual_prefix`), jamais réutilisé ici."""
    path = tree.get_path(node_id)
    parts = []
    for node in path:
        if node.type in (NodeType.DOCUMENT, NodeType.SECTION, NodeType.SUBSECTION, NodeType.HEADING):
            title = node.metadata.get("title") or node.text or node.markdown or node.type.value
            parts.append(title.strip())
    return " > ".join(parts) if parts else "Document"


def get_structural_ancestors(tree: DocumentTree, node_id) -> list[DOMNode]:
    """Ancêtres qui sont des nœuds structurels (section, heading, document)."""
    return [n for n in get_ancestors(tree, node_id) if n.is_structural]


def derive_parent_path(hierarchy_path: str) -> str | None:
    """Clé de groupe de fratrie, dérivée uniquement de `hierarchy_path` — pas
    de `parent_chunk_id` : ce champ n'est peuplé que pour les sources
    chunkées hiérarchiquement (Markdown), jamais pour les sources
    structurées JSON/YAML (ex. specs OpenAPI), là où ce mécanisme est
    justement nécessaire (une liste JSON — ex. les paramètres d'un endpoint —
    devient un chunk par élément : `...parameters[0]`, `...parameters[1]`...,
    qui doivent pouvoir être retrouvés ensemble même si le retrieval n'en
    classe qu'une partie dans le top-k).

    Retire le suffixe `[N]` final : deux chunks partageant le même
    `parent_path` sont des éléments de la même liste source. Retourne None
    si `hierarchy_path` ne se termine pas par un tel index — un chunk qui
    n'est pas un élément de liste énumérée n'appartient à aucun groupe de
    fratrie, jamais regroupé avec un autre par coïncidence de chemin.

    Générique par construction : ne dépend que de la structure du chemin
    (propriété du chunking, pas du domaine documentaire) — s'applique
    identiquement à n'importe quelle source, présente ou future.
    """
    if _TRAILING_INDEX_RE.search(hierarchy_path or ""):
        return _TRAILING_INDEX_RE.sub("", hierarchy_path)
    return None
