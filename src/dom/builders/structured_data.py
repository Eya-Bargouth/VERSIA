"""Builder générique pour données structurées (YAML/JSON) — parcours
récursif générique uniquement, quelle que soit la forme du document.

Aucune extraction spécifique à un schéma (ex. OpenAPI) : ce builder ne
connaît que la structure syntaxique (dict/liste/scalaire) et une seule
heuristique de taille, jamais la sémantique d'un domaine documentaire
particulier — cohérent avec l'invariant du projet ("couplage au format,
jamais à la sémantique d'un corpus donné") et avec le sujet de stage
("architecture indépendante du domaine documentaire", "réutilisable sans
réécriture majeure").

Règle de décision, à chaque nœud (dict ou liste) :
- Liste de dicts → un enregistrement par élément (feuille), jamais fragmenté
  plus loin (le contenu imbriqué d'un élément, ex. incident_updates, reste
  inclus dans son chunk).
- Dict "conteneur pur" (TOUTES ses valeurs sont elles-mêmes des collections
  non vides — dict ou liste de dicts, ex. {"paths": {"/v1/orders": {...}}})
  → continue à descendre pour trouver les vrais enregistrements.
- Dict "mixte" (mélange de scalaires et de structures, ex. un endpoint
  OpenAPI avec summary/parameters/responses) → traité comme UN seul
  enregistrement (dumpé entier, y compris ses paramètres), SAUF s'il dépasse
  un seuil de taille (_LEAF_SIZE_THRESHOLD) — auquel cas il est quand même
  décomposé clé par clé pour éviter un chunk disproportionné. Seuil
  générique de taille, pas une règle de domaine.

Historique : une première version de ce builder détectait spécifiquement la
forme OpenAPI (paths/méthodes HTTP) pour produire des types de nœuds riches
(API_ENDPOINT/API_PARAMETER/API_RESPONSE). Elle a été retirée : ce chemin
spécialisé fragmentait un endpoint et ses paramètres en nœuds séparés sans
jamais les refusionner dans un même chunk (bug réel trouvé en Phase 5 — les
paramètres requis d'un endpoint disparaissaient du corpus indexé). La règle
"dict mixte = un seul enregistrement" ci-dessus résout ce cas précis sans
code spécifique à OpenAPI : un endpoint est un dict mixte comme un autre.
"""

import json
from abc import abstractmethod
from typing import Any

from src.config.source_config import SourceConfig
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import DOMNode, DocumentTree, NodeType

# Taille max (caractères JSON sérialisés) d'un dict "mixte" avant qu'il ne
# soit quand même décomposé clé par clé plutôt que dumpé entier. Générique —
# évite qu'un enregistrement disproportionné (ex. un endpoint à la
# description très longue) devienne un chunk inexploitable, sans référence
# à aucun domaine documentaire particulier.
_LEAF_SIZE_THRESHOLD = 2000


class StructuredDataBuilder(AbstractDOMBuilder):
    """Base commune YAML/JSON — ne diffère que par le chargement du fichier."""

    @abstractmethod
    def _load(self, source_path: str) -> Any:
        """Parse le fichier en structure Python native (dict/list/scalaire)."""
        ...

    def build(self, source_path: str, config: SourceConfig) -> DocumentTree:
        data = self._load(source_path)

        root = self._create_root_node(source_path, config)
        tree = DocumentTree(root_id=root.id, source_id=config.source_id, source_path=source_path)
        tree.add_node(root)

        self._walk_node(data, config, source_path, tree, root.id, path="")

        tree.compute_all_hashes()
        return tree

    # ------------------------------------------------------------------
    # Parcours générique
    # ------------------------------------------------------------------

    def _walk_node(self, value: Any, config: SourceConfig, source_path: str, tree: DocumentTree, parent_id, path: str) -> None:
        if isinstance(value, dict) and value and self._should_descend_dict(value):
            for key, sub in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                section = self._add_section(str(key), config, source_path, tree, parent_id)
                self._walk_node(sub, config, source_path, tree, section.id, child_path)
            return

        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
            section = self._add_section(path or "records", config, source_path, tree, parent_id)
            for idx, item in enumerate(value):
                title = f"{path}[{idx}]" if path else f"item_{idx}"
                self._add_document_leaf(item, config, source_path, tree, section.id, title=title)
            return

        # Enregistrement (dict mixte, scalaire, ou liste de non-dicts) —
        # feuille, dumpé entier, jamais fragmenté plus loin.
        self._add_document_leaf(value, config, source_path, tree, parent_id, title=path or "document")

    def _should_descend_dict(self, value: dict) -> bool:
        if self._is_pure_container(value):
            return True
        # Dict mixte : ne descend que s'il est trop volumineux pour rester
        # un seul chunk (seuil générique, voir _LEAF_SIZE_THRESHOLD).
        try:
            size = len(json.dumps(value, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            size = 0
        return size > _LEAF_SIZE_THRESHOLD

    @staticmethod
    def _is_pure_container(value: dict) -> bool:
        """True si TOUTES les valeurs sont elles-mêmes des collections non
        vides (dict ou liste de dicts) — un conteneur d'enregistrements
        nommés (ex. "paths", ou "channels"), pas un enregistrement lui-même."""
        return all(StructuredDataBuilder._is_collection_value(v) for v in value.values())

    @staticmethod
    def _is_collection_value(v: Any) -> bool:
        if isinstance(v, dict) and v:
            return True
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            return True
        return False

    def _add_section(self, title: str, config: SourceConfig, source_path: str, tree: DocumentTree, parent_id) -> DOMNode:
        node = DOMNode(
            type=NodeType.SECTION,
            source_id=config.source_id,
            source_path=source_path,
            text=title,
            markdown=f"## {title}",
            metadata={"title": title},
            is_structural=True,
            is_content=False,
        )
        tree.add_node(node, parent_id=parent_id)
        return node

    def _add_document_leaf(self, item: Any, config: SourceConfig, source_path: str, tree: DocumentTree, parent_id, title: str) -> DOMNode:
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, indent=2, default=str)
        markdown = text if isinstance(item, str) else f"```json\n{text}\n```"
        keys = list(item.keys()) if isinstance(item, dict) else []
        node = DOMNode(
            type=NodeType.DOCUMENT,
            source_id=config.source_id,
            source_path=source_path,
            text=text,
            markdown=markdown,
            metadata={"title": title, "keys": keys},
            is_structural=False,
            is_content=True,
        )
        tree.add_node(node, parent_id=parent_id)
        return node
