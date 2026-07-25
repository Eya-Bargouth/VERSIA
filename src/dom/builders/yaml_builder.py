"""Builder YAML structuré (OpenAPI specs) — parse la structure native, jamais texte brut."""

from pathlib import Path
from uuid import uuid4

import yaml

from src.config.manifest_schema import SourceManifest
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import DOMNode, DocumentTree, NodeType


class YAMLBuilder(AbstractDOMBuilder):
    """Builder pour fichiers YAML structurés (OpenAPI, JSON Schema)."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".yaml", ".yml"}

    def build(self, source_path: str, manifest: SourceManifest) -> DocumentTree:
        with open(source_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        root = self._create_root_node(source_path, manifest.source_id)
        tree = DocumentTree(root_id=root.id, source_id=manifest.source_id, source_path=source_path)
        tree.add_node(root)

        # Nœud info si présent
        if isinstance(data, dict) and "info" in data:
            info_node = DOMNode(
                type=NodeType.METADATA,
                source_id=manifest.source_id,
                source_path=source_path,
                text=str(data["info"]),
                markdown=f"```yaml\n{yaml.dump(data['info'])}\n```",
                is_structural=False,
            )
            tree.add_node(info_node, parent_id=root.id)

        # Nœud paths
        if isinstance(data, dict) and "paths" in data:
            paths_node = DOMNode(
                type=NodeType.SECTION,
                source_id=manifest.source_id,
                source_path=source_path,
                text="paths",
                markdown="## paths",
                is_structural=True,
                is_content=False,
            )
            tree.add_node(paths_node, parent_id=root.id)

            for path_str, methods in data["paths"].items():
                if not isinstance(methods, dict):
                    continue
                for method, spec in methods.items():
                    if not isinstance(spec, dict):
                        continue
                    endpoint_node = self._build_endpoint_node(
                        path_str, method, spec, manifest, source_path, paths_node.id
                    )
                    tree.add_node(endpoint_node, parent_id=paths_node.id)

                    # Paramètres
                    params = spec.get("parameters", [])
                    for p in params:
                        param_node = self._build_parameter_node(p, manifest, source_path, endpoint_node.id)
                        tree.add_node(param_node, parent_id=endpoint_node.id)

                    # Réponses
                    responses = spec.get("responses", {})
                    for code, resp in responses.items():
                        resp_node = self._build_response_node(code, resp, manifest, source_path, endpoint_node.id)
                        tree.add_node(resp_node, parent_id=endpoint_node.id)

        tree.compute_all_hashes()
        return tree

    def _build_endpoint_node(self, path: str, method: str, spec: dict, manifest, source_path: str, parent_id):
        summary = spec.get("summary", "")
        description = spec.get("description", "")
        md_lines = [
            f"### {method.upper()} {path}",
            "",
            f"**Summary:** {summary}" if summary else "",
            f"**Description:** {description}" if description else "",
            "",
        ]
        md = "\n".join(line for line in md_lines if line)

        return DOMNode(
            type=NodeType.API_ENDPOINT,
            source_id=manifest.source_id,
            source_path=source_path,
            text=f"{method.upper()} {path} — {summary}",
            markdown=md,
            metadata={"method": method, "path": path, "summary": summary, "tags": spec.get("tags", [])},
            is_structural=False,
            is_content=True,
            parent_id=parent_id,
        )

    def _build_parameter_node(self, param: dict, manifest, source_path: str, parent_id):
        name = param.get("name", "unknown")
        ptype = param.get("schema", {}).get("type", param.get("type", "unknown"))
        required = param.get("required", False)
        desc = param.get("description", "")

        md = f"- **{name}** ({ptype}, {'required' if required else 'optional'}): {desc}"
        return DOMNode(
            type=NodeType.API_PARAMETER,
            source_id=manifest.source_id,
            source_path=source_path,
            text=f"Parameter {name}: {ptype}",
            markdown=md,
            metadata={"name": name, "type": ptype, "required": required, "in": param.get("in", "")},
            parent_id=parent_id,
        )

    def _build_response_node(self, code: str, resp: dict, manifest, source_path: str, parent_id):
        desc = resp.get("description", "")
        md = f"- **{code}**: {desc}"
        return DOMNode(
            type=NodeType.API_RESPONSE,
            source_id=manifest.source_id,
            source_path=source_path,
            text=f"Response {code}: {desc}",
            markdown=md,
            metadata={"code": code, "description": desc},
            parent_id=parent_id,
        )
