"""Builder JSON avec règles de regroupement configurables."""

import json
from pathlib import Path

from src.config.manifest_schema import SourceManifest
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import DOMNode, DocumentTree, NodeType


class JSONBuilder(AbstractDOMBuilder):
    """Builder pour fichiers JSON — group_nested extrait des sous-documents."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".json"

    def build(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = self._create_root_node(source_path, manifest)
        tree = DocumentTree(root_id=root.id, source_id=manifest.source_id, source_path=source_path)
        tree.add_node(root)

        group_keys = manifest.chunking_policy.group_nested

        if isinstance(data, list) and group_keys:
            # Liste d'objets, chaque objet est un document
            for idx, item in enumerate(data):
                doc_node = self._build_json_document_node(item, idx, manifest, source_path, root.id, group_keys)
                tree.add_node(doc_node, parent_id=root.id)
        elif isinstance(data, dict) and group_keys:
            # Objet racine avec clés groupées
            for key in group_keys:
                if key in data and isinstance(data[key], list):
                    section = DOMNode(
                        type=NodeType.SECTION,
                        source_id=manifest.source_id,
                        source_path=source_path,
                        text=key,
                        markdown=f"## {key}",
                        is_structural=True,
                        is_content=False,
                    )
                    tree.add_node(section, parent_id=root.id)
                    for idx, item in enumerate(data[key]):
                        doc_node = self._build_json_document_node(item, idx, manifest, source_path, section.id, [])
                        tree.add_node(doc_node, parent_id=section.id)
                elif key in data:
                    doc_node = self._build_json_document_node(data[key], 0, manifest, source_path, root.id, [])
                    tree.add_node(doc_node, parent_id=root.id)
        else:
            # Document unique
            doc_node = self._build_json_document_node(data, 0, manifest, source_path, root.id, [])
            tree.add_node(doc_node, parent_id=root.id)

        tree.compute_all_hashes()
        return tree

    def _build_json_document_node(self, item, index: int, manifest, source_path: str, parent_id, group_keys: list):
        text = json.dumps(item, ensure_ascii=False, indent=2)
        title = item.get("title", item.get("name", f"item_{index}")) if isinstance(item, dict) else f"item_{index}"

        return DOMNode(
            type=NodeType.DOCUMENT,
            source_id=manifest.source_id,
            source_path=source_path,
            text=text,
            markdown=f"```json\n{text}\n```",
            metadata={"title": title, "index": index, "keys": list(item.keys()) if isinstance(item, dict) else []},
            is_structural=False,
            is_content=True,
            parent_id=parent_id,
        )
