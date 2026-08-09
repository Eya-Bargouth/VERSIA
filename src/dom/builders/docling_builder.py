"""Builder PDF/DOCX via Docling avec fallback pdfplumber/unstructured."""

import logging
from pathlib import Path
from uuid import uuid4

from src.config.manifest_schema import SourceManifest
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import BoundingBox, DOMNode, DocumentTree, NodeType, TextStyle

logger = logging.getLogger(__name__)


class DoclingBuilder(AbstractDOMBuilder):
    """Builder pour PDF/DOCX/PPTX. Fallback sur pdfplumber/unstructured si Docling échoue."""

    def supports(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in {".pdf", ".docx", ".pptx"}

    def build(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        try:
            return self._build_with_docling(source_path, manifest)
        except Exception as exc:
            logger.warning("docling_failed", error=str(exc), fallback="pdfplumber/unstructured")
            ext = Path(source_path).suffix.lower()
            if ext == ".pdf":
                return self._build_with_pdfplumber(source_path, manifest)
            raise RuntimeError(f"No fallback available for {ext}")

    def _build_with_docling(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError("Docling not installed")

        converter = DocumentConverter()
        result = converter.convert(source_path)
        doc = result.document

        root = self._create_root_node(source_path, manifest)
        tree = DocumentTree(root_id=root.id, source_id=manifest.source_id, source_path=source_path)
        tree.add_node(root)

        # Itération simplifiée sur les éléments exportés en markdown
        for item in doc.iterate_items():
            node = self._docling_item_to_node(item, manifest.source_id, source_path, root.id)
            tree.add_node(node, parent_id=root.id)

        tree.compute_all_hashes()
        return tree

    def _docling_item_to_node(self, item, source_id: str, source_path: str, parent_id):
        # Mapping simplifié — Docling API peut varier selon la version
        # On suppose item a .label et .text
        label = getattr(item, "label", None)
        text = getattr(item, "text", "") or ""

        type_map = {
            "paragraph": NodeType.PARAGRAPH,
            "heading": NodeType.HEADING,
            "table": NodeType.TABLE,
            "list": NodeType.LIST,
            "list_item": NodeType.LIST_ITEM,
            "figure": NodeType.FIGURE,
            "caption": NodeType.CAPTION,
            "code": NodeType.CODE_BLOCK,
        }
        node_type = type_map.get(str(label).lower(), NodeType.PARAGRAPH)

        bbox = None
        if hasattr(item, "bbox") and item.bbox:
            bbox = BoundingBox(
                x=item.bbox.l,
                y=item.bbox.t,
                width=item.bbox.r - item.bbox.l,
                height=item.bbox.b - item.bbox.t,
                page=getattr(item, "page", None),
            )

        return DOMNode(
            type=node_type,
            source_id=source_id,
            source_path=source_path,
            text=text,
            markdown=text,
            parent_id=parent_id,
            is_structural=node_type in (NodeType.HEADING, NodeType.SECTION),
            is_content=node_type not in (NodeType.DOCUMENT, NodeType.SECTION, NodeType.HEADING),
            bbox=bbox,
        )

    def _build_with_pdfplumber(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        import pdfplumber

        root = self._create_root_node(source_path, manifest)
        tree = DocumentTree(root_id=root.id, source_id=manifest.source_id, source_path=source_path)
        tree.add_node(root)

        with pdfplumber.open(source_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_node = DOMNode(
                    type=NodeType.PAGE,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    page_num=page_num,
                    is_structural=True,
                    is_content=False,
                )
                tree.add_node(page_node, parent_id=root.id)

                words = page.extract_words()
                if words:
                    lines = self._words_to_lines(words)
                    for line_text in lines:
                        para = DOMNode(
                            type=NodeType.PARAGRAPH,
                            source_id=manifest.source_id,
                            source_path=source_path,
                            text=line_text,
                            markdown=line_text,
                            page_num=page_num,
                        )
                        tree.add_node(para, parent_id=page_node.id)
                else:
                    text = page.extract_text() or ""
                    for line in text.split("\n"):
                        if line.strip():
                            para = DOMNode(
                                type=NodeType.PARAGRAPH,
                                source_id=manifest.source_id,
                                source_path=source_path,
                                text=line,
                                markdown=line,
                                page_num=page_num,
                            )
                            tree.add_node(para, parent_id=page_node.id)

        tree.compute_all_hashes()
        return tree

    @staticmethod
    def _words_to_lines(words: list[dict]) -> list[str]:
        """Regroupe les mots par ligne approximative (top coordinate)."""
        if not words:
            return []
        lines: dict[int, list[str]] = {}
        for w in words:
            top = round(float(w.get("top", 0)) / 5) * 5  # bucket 5px
            lines.setdefault(top, []).append(w.get("text", ""))
        return [" ".join(lines[k]) for k in sorted(lines.keys())]
