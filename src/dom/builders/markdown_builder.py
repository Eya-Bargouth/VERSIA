"""Builder Markdown via markdown-it-py."""

from pathlib import Path

from markdown_it import MarkdownIt

from src.config.manifest_schema import SourceManifest
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import DOMNode, DocumentTree, NodeType


class MarkdownBuilder(AbstractDOMBuilder):
    """Builder pour fichiers Markdown — préserve la hiérarchie des headings."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".md", ".markdown"}

    def build(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        with open(source_path, "r", encoding="utf-8") as f:
            text = f.read()

        md = MarkdownIt()
        tokens = md.parse(text)

        root = self._create_root_node(source_path, manifest.source_id)
        tree = DocumentTree(root_id=root.id, source_id=manifest.source_id, source_path=source_path)
        tree.add_node(root)

        # Stack pour gérer la hiérarchie des sections
        stack = [(root.id, 0)]  # (node_id, heading_level)
        current_parent = root.id

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                level = int(token.tag.replace("h", ""))
                # Récupérer le texte du heading
                i += 1
                inline = tokens[i]
                title = inline.content if inline.type == "inline" else ""
                i += 1  # heading_close

                # Remonter la stack jusqu'au niveau parent approprié
                while stack and stack[-1][1] >= level:
                    stack.pop()
                parent_id = stack[-1][0] if stack else root.id

                heading_node = DOMNode(
                    type=NodeType.HEADING,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    text=title,
                    markdown=f"{'#' * level} {title}",
                    metadata={"level": level, "title": title},
                    is_structural=True,
                    is_content=False,
                    level=level,
                )
                tree.add_node(heading_node, parent_id=parent_id)
                stack.append((heading_node.id, level))
                current_parent = heading_node.id

            elif token.type == "paragraph_open":
                i += 1
                inline = tokens[i]
                content = inline.content if inline.type == "inline" else ""
                i += 1  # paragraph_close
                para = DOMNode(
                    type=NodeType.PARAGRAPH,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    text=content,
                    markdown=content,
                )
                tree.add_node(para, parent_id=current_parent)

            elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
                list_node = DOMNode(
                    type=NodeType.LIST,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    text="",
                    markdown="",
                    is_structural=True,
                    is_content=False,
                )
                tree.add_node(list_node, parent_id=current_parent)
                current_parent = list_node.id
                i += 1

            elif token.type == "list_item_open":
                i += 1
                inline = tokens[i]
                content = inline.content if inline.type == "inline" else ""
                i += 1  # list_item_close
                item = DOMNode(
                    type=NodeType.LIST_ITEM,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    text=content,
                    markdown=f"- {content}",
                )
                tree.add_node(item, parent_id=current_parent)

            elif token.type == "bullet_list_close" or token.type == "ordered_list_close":
                stack = [(nid, lvl) for nid, lvl in stack if tree.get_node(nid) and tree.get_node(nid).type != NodeType.LIST]
                current_parent = stack[-1][0] if stack else root.id
                i += 1

            elif token.type == "fence":
                code = DOMNode(
                    type=NodeType.CODE_BLOCK,
                    source_id=manifest.source_id,
                    source_path=source_path,
                    text=token.content,
                    markdown=f"```{token.info}\n{token.content}\n```",
                    metadata={"language": token.info},
                )
                tree.add_node(code, parent_id=current_parent)
                i += 1

            else:
                i += 1

        tree.compute_all_hashes()
        return tree
