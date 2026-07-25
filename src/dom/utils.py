"""Helpers de navigation et manipulation de l'arbre DOM."""

from src.dom.models import DOMNode, DocumentTree, NodeType


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
    """Construit le fil d'Ariane : 'Document > Section 2 > Subsection 2.1'."""
    path = tree.get_path(node_id)
    parts = []
    for node in path:
        if node.type in (NodeType.DOCUMENT, NodeType.SECTION, NodeType.SUBSECTION, NodeType.HEADING):
            title = node.metadata.get("title") or node.text or node.markdown or node.type.value
            parts.append(title.strip()[:60])
    return " > ".join(parts) if parts else "Document"


def get_structural_ancestors(tree: DocumentTree, node_id) -> list[DOMNode]:
    """Ancêtres qui sont des nœuds structurels (section, heading, document)."""
    return [n for n in get_ancestors(tree, node_id) if n.is_structural]
