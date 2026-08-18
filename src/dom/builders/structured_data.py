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
import re
from abc import abstractmethod
from typing import Any

import structlog

from src.config.source_config import SourceConfig
from src.dom.builders.base import AbstractDOMBuilder
from src.dom.models import DOMNode, DocumentTree, NodeType

logger = structlog.get_logger(__name__)

# Taille max (caractères JSON sérialisés) d'un dict "mixte" avant qu'il ne
# soit quand même décomposé clé par clé plutôt que dumpé entier. Générique —
# évite qu'un enregistrement disproportionné (ex. un endpoint à la
# description très longue) devienne un chunk inexploitable, sans référence
# à aucun domaine documentaire particulier.
_LEAF_SIZE_THRESHOLD = 2000

# Profondeur max de résolution $ref en chaîne (ref -> cible contenant elle-
# même un ref -> ...). Un document au graphe de schémas densément
# interconnecté (composants qui se référencent tous mutuellement, cas réel
# constaté sur un spec OpenAPI volumineux — ex. PaymentIntent -> Charge ->
# Customer -> PaymentMethod -> ...) produit une explosion combinatoire à la
# sérialisation JSON si on résout en chaîne sans limite stricte : un objet
# partagé référencé depuis N endroits, chacun référencé depuis M autres, doit
# être dupliqué en toutes lettres à chaque occurrence (JSON n'a pas de notion
# de partage), ce qui fait exploser la taille du texte de façon combinatoire
# avec la profondeur.
#
# Pas de limite fixe unique : un document dont les $ref pointent surtout vers
# une poignée de cibles partagées (ex. un paramètre référencé par de
# nombreux endpoints — beaucoup d'occurrences, peu de cibles distinctes) n'a
# pas ce risque et peut être résolu plus profondément sans danger. Seul un
# document où les cibles sont majoritairement distinctes (signe d'un web de
# composants réellement interconnectés, chacun pouvant à son tour référencer
# autre chose) doit rester restreint à un seul niveau. La densité est donc
# mesurée sur le document lui-même (ratio cibles $ref uniques / occurrences
# totales, voir _estimate_max_ref_depth) plutôt que fixée a priori — mesure
# structurelle générique (propriété du format, pas de la sémantique d'un
# corpus donné), calibrée empiriquement sur les deux cas réels rencontrés :
# ratio ~0.08 (beaucoup de réutilisation, sûr) vs ~0.37 (interconnexion
# dense, risqué).
_REF_DEPTH_DENSITY_THRESHOLD = 0.15
_MAX_REF_DEPTH_DENSE = 1
_MAX_REF_DEPTH_SPARSE = 3


class StructuredDataBuilder(AbstractDOMBuilder):
    """Base commune YAML/JSON — ne diffère que par le chargement du fichier."""

    @abstractmethod
    def _load(self, source_path: str) -> Any:
        """Parse le fichier en structure Python native (dict/list/scalaire)."""
        ...

    def build(self, source_path: str, config: SourceConfig) -> DocumentTree:
        data = self._load(source_path)
        self._ref_root = data
        self._ref_cache: dict[str, Any] = {}
        self._max_ref_depth = self._estimate_max_ref_depth(data)

        root = self._create_root_node(source_path, config)
        tree = DocumentTree(root_id=root.id, source_id=config.source_id, source_path=source_path)
        tree.add_node(root)

        self._walk_node(data, config, source_path, tree, root.id, path="")

        tree.compute_all_hashes()
        return tree

    # ------------------------------------------------------------------
    # Résolution $ref (JSON Pointer interne au document) — mécanique
    # générique du format YAML/JSON, pas une notion OpenAPI : un
    # enregistrement référencé (ex. un paramètre partagé) doit apparaître
    # dans le même chunk que l'enregistrement qui le référence, plutôt que
    # de nécessiter deux chunks liés pour être compris.
    #
    # Appliquée localement (sur un seul nœud à la fois, au moment où il est
    # examiné par le parcours), jamais en un seul pré-passage sur le
    # document entier : un document dont les schémas se référencent
    # massivement les uns les autres (composants tous interconnectés, cas
    # réel constaté sur un spec OpenAPI volumineux) ferait sinon reconstruire
    # récursivement la totalité du document une fois de plus rien que pour
    # la résolution, en plus du parcours normal — coûteux sans bénéfice,
    # puisque seuls les nœuds "feuille" (petits, déjà isolés par le parcours
    # générique avant d'être sérialisés) ont réellement besoin d'être
    # résolus. `_ref_cache` reste mémoïsé sur toute la durée d'un build().
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_max_ref_depth(data: Any) -> int:
        """Sonde la densité du graphe $ref du document (une passe, avant
        toute résolution) pour choisir la profondeur max : ratio entre le
        nombre de cibles $ref distinctes et le nombre total d'occurrences.
        Coût borné à une sérialisation du document brut (pas de résolution
        récursive ici) — sans rapport avec l'explosion combinatoire que la
        limite de profondeur elle-même prévient."""
        try:
            text = json.dumps(data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return _MAX_REF_DEPTH_DENSE
        refs = re.findall(r'"\$ref"\s*:\s*"(#/[^"]+)"', text)
        if not refs:
            return _MAX_REF_DEPTH_SPARSE
        density = len(set(refs)) / len(refs)
        return _MAX_REF_DEPTH_DENSE if density > _REF_DEPTH_DENSITY_THRESHOLD else _MAX_REF_DEPTH_SPARSE

    def _resolve_refs(self, value: Any, source_path: str) -> Any:
        """Remplace récursivement tout {"$ref": "#/a/b/c"} par le sous-arbre
        pointé (JSON Pointer, RFC 6901) à l'intérieur de `value` seulement.
        Seules les références internes au document ("#/...") sont résolues —
        une référence vers un autre fichier est laissée telle quelle."""

        def resolve_pointer(ref: str) -> Any:
            node = self._ref_root
            for raw_part in ref[2:].split("/"):
                part = raw_part.replace("~1", "/").replace("~0", "~")
                node = node[int(part)] if isinstance(node, list) else node[part]
            return node

        def walk(node: Any, seen: frozenset, depth: int) -> Any:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/"):
                    if ref in self._ref_cache:
                        return self._ref_cache[ref]
                    if ref in seen:
                        logger.warning("ref_cycle_detected", source_path=source_path, ref=ref)
                        return node
                    if depth >= self._max_ref_depth:
                        logger.warning("ref_depth_limit_reached", source_path=source_path, ref=ref, depth=depth)
                        return node
                    try:
                        target = resolve_pointer(ref)
                    except (KeyError, IndexError, TypeError, ValueError):
                        logger.warning("ref_resolution_failed", source_path=source_path, ref=ref)
                        return node
                    resolved = walk(target, seen | {ref}, depth + 1)
                    self._ref_cache[ref] = resolved
                    return resolved
                return {key: walk(sub, seen, depth) for key, sub in node.items()}
            if isinstance(node, list):
                return [walk(item, seen, depth) for item in node]
            return node

        return walk(value, frozenset(), 0)

    # ------------------------------------------------------------------
    # Parcours générique
    # ------------------------------------------------------------------

    def _walk_node(self, value: Any, config: SourceConfig, source_path: str, tree: DocumentTree, parent_id, path: str) -> None:
        # Substitution $ref au niveau conteneur (rare mais possible, ex. un
        # chemin entier délégué via $ref) — bornée à ce seul nœud, avant de
        # décider comment le traiter ; les $ref imbriqués plus profondément
        # restent résolus paresseusement, au moment où le parcours les atteint.
        if isinstance(value, dict) and set(value.keys()) == {"$ref"}:
            value = self._resolve_refs(value, source_path)

        if isinstance(value, dict) and value and self._should_descend_dict(value, source_path):
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
        # feuille, dumpé entier, jamais fragmenté plus loin. Nœud borné
        # (déjà isolé par le parcours ci-dessus) : c'est ici, sur un
        # fragment de taille raisonnable plutôt que sur le document entier,
        # que la résolution récursive complète des $ref est appliquée.
        resolved = self._resolve_refs(value, source_path) if isinstance(value, (dict, list)) else value
        self._add_document_leaf(resolved, config, source_path, tree, parent_id, title=path or "document")

    def _should_descend_dict(self, value: dict, source_path: str) -> bool:
        if self._is_pure_container(value):
            return True
        # Dict mixte : ne descend que s'il est trop volumineux pour rester
        # un seul chunk (seuil générique, voir _LEAF_SIZE_THRESHOLD). $ref
        # résolus ici (nœud déjà borné, pas le document entier) pour que la
        # taille mesurée reflète le contenu réel, pas un pointeur brut.
        resolved = self._resolve_refs(value, source_path)
        try:
            size = len(json.dumps(resolved, ensure_ascii=False, default=str))
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
