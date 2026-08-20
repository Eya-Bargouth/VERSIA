"""Tests unitaires pour le DOM unifié."""

import pytest
from uuid import uuid4

from src.dom.models import BoundingBox, DOMNode, DocumentTree, NodeType, TextStyle
from src.dom.utils import build_hierarchy_path, derive_parent_path, get_ancestors, get_descendants, get_structural_ancestors


pytestmark = pytest.mark.phase1


class TestDOMNode:
    def test_node_creation(self):
        node = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")
        assert node.type == NodeType.PARAGRAPH
        assert node.source_id == "test"
        assert node.id is not None

    def test_compute_hash(self):
        node = DOMNode(
            type=NodeType.PARAGRAPH,
            source_id="test",
            source_path="/test.md",
            markdown="Hello world",
        )
        h = node.compute_hash()
        assert h is not None
        assert len(h) == 64  # SHA-256 hex

    def test_no_self_reference(self):
        nid = uuid4()
        with pytest.raises(ValueError, match="own child"):
            DOMNode(
                id=nid,
                type=NodeType.PARAGRAPH,
                source_id="test",
                source_path="/test.md",
                children_ids=[nid],
            )


class TestDocumentTree:
    def test_tree_creation(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        assert tree.root_id == root.id
        assert len(tree.nodes) == 1

    def test_add_child(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)

        child = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")
        tree.add_node(child, parent_id=root.id)

        assert child.parent_id == root.id
        assert child.id in [c.id for c in tree.get_children(root.id)]

    def test_get_path(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        sec = DOMNode(type=NodeType.SECTION, source_id="test", source_path="/test.md")
        para = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(sec, parent_id=root.id)
        tree.add_node(para, parent_id=sec.id)

        path = tree.get_path(para.id)
        assert len(path) == 3
        assert path[0].type == NodeType.DOCUMENT
        assert path[1].type == NodeType.SECTION
        assert path[2].type == NodeType.PARAGRAPH

    def test_get_siblings(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        c1 = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")
        c2 = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(c1, parent_id=root.id)
        tree.add_node(c2, parent_id=root.id)

        siblings = tree.get_siblings(c1.id)
        assert len(siblings) == 1
        assert siblings[0].id == c2.id

    def test_integrity_validation(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)

        bad = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md", parent_id=uuid4())
        tree.add_node(bad)

        errors = tree.validate_integrity()
        assert len(errors) >= 1
        assert "missing parent" in errors[0].lower()

    def test_compute_all_hashes(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        child = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md", text="hello")
        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(child, parent_id=root.id)
        tree.compute_all_hashes()
        assert child.content_hash is not None


class TestDOMUtils:
    def test_build_hierarchy_path(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md", metadata={"title": "Doc"})
        sec = DOMNode(type=NodeType.SECTION, source_id="test", source_path="/test.md", metadata={"title": "Sec 1"})
        para = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(sec, parent_id=root.id)
        tree.add_node(para, parent_id=sec.id)

        path = build_hierarchy_path(tree, para.id)
        assert "Doc" in path
        assert "Sec 1" in path

    def test_derive_parent_path_strips_trailing_index(self):
        """Cas réel : 4 éléments d'une liste JSON (ex. paramètres d'un
        endpoint) chunkés séparément doivent partager le même parent_path,
        pour pouvoir être retrouvés ensemble (voir hybrid_retriever.py::
        _fetch_missing_siblings)."""
        base = "document > paths > /v1/charges/search > get > parameters"
        assert derive_parent_path(f"{base}[0]") == base
        assert derive_parent_path(f"{base}[1]") == base
        assert derive_parent_path(f"{base}[12]") == base

    def test_derive_parent_path_none_without_index(self):
        assert derive_parent_path("document > paths > /v1/charges/search > get > summary") is None

    def test_derive_parent_path_handles_empty(self):
        assert derive_parent_path("") is None
        assert derive_parent_path(None) is None

    def test_get_ancestors(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        sec = DOMNode(type=NodeType.SECTION, source_id="test", source_path="/test.md")
        para = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(sec, parent_id=root.id)
        tree.add_node(para, parent_id=sec.id)

        ancestors = get_ancestors(tree, para.id)
        assert len(ancestors) == 2
        assert ancestors[0].id == root.id
        assert ancestors[1].id == sec.id

    def test_get_descendants(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md")
        sec = DOMNode(type=NodeType.SECTION, source_id="test", source_path="/test.md")
        p1 = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")
        p2 = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(sec, parent_id=root.id)
        tree.add_node(p1, parent_id=sec.id)
        tree.add_node(p2, parent_id=sec.id)

        desc = get_descendants(tree, root.id)
        assert len(desc) == 3

    def test_get_structural_ancestors(self):
        root = DOMNode(type=NodeType.DOCUMENT, source_id="test", source_path="/test.md", is_structural=True)
        sec = DOMNode(type=NodeType.SECTION, source_id="test", source_path="/test.md", is_structural=True)
        para = DOMNode(type=NodeType.PARAGRAPH, source_id="test", source_path="/test.md")

        tree = DocumentTree(root_id=root.id, source_id="test", source_path="/test.md")
        tree.add_node(root)
        tree.add_node(sec, parent_id=root.id)
        tree.add_node(para, parent_id=sec.id)

        struct = get_structural_ancestors(tree, para.id)
        assert len(struct) == 2
        assert struct[0].type == NodeType.DOCUMENT
        assert struct[1].type == NodeType.SECTION