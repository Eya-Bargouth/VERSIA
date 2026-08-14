"""Tests unitaires pour les DOM builders."""

import pytest

from src.config.source_config import SourceConfig
from src.dom.builders.json_builder import JSONBuilder
from src.dom.builders.markdown_builder import MarkdownBuilder
from src.dom.builders.yaml_builder import YAMLBuilder
from src.dom.models import NodeType

pytestmark = pytest.mark.phase1


class TestRootNodeVersioning:
    """AbstractDOMBuilder._create_root_node doit appliquer
    config.version_pattern à la racine — gap Phase 2 corrigé en Phase 4
    (voir docs/PHASE_4_SUMMARY.md), toujours vrai après le passage à
    SourceConfig (Phase 5 — plus de manifeste, override minimal)."""

    def test_filename_pattern_sets_version_tag_and_order(self, tmp_path):
        v_path = tmp_path / "spec3-v2213.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\npaths: {}\n",
            encoding="utf-8",
        )
        config = SourceConfig(
            source_id="test_api",
            version_pattern=r"spec3-(?P<version>.+)\.yaml",
            version_order=["legacy", "v2213", "v2293"],
        )
        tree = YAMLBuilder().build(str(v_path), config)
        root = tree.nodes[tree.root_id]
        assert root.version_tag == "v2213"
        assert root.version_order == 1

    def test_no_versioning_pattern_leaves_root_untagged(self, sample_openapi_path):
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        root = tree.nodes[tree.root_id]
        assert root.version_tag is None
        assert root.version_order is None

    def test_pattern_not_matching_filename_does_not_crash(self, tmp_path):
        v_path = tmp_path / "unrelated_name.yaml"
        v_path.write_text(
            "openapi: \"3.0.3\"\ninfo:\n  title: X\n  version: \"1.0.0\"\npaths: {}\n",
            encoding="utf-8",
        )
        config = SourceConfig(
            source_id="test_api",
            version_pattern=r"spec3-(?P<version>.+)\.yaml",
            version_order=["v1"],
        )
        tree = YAMLBuilder().build(str(v_path), config)
        assert tree.nodes[tree.root_id].version_tag is None

    def test_status_always_active_by_default(self, sample_openapi_path):
        """La validité par source (ancien ValidityConfig) a été supprimée —
        toute source est "active" par défaut, sans exception déclarable."""
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        assert tree.nodes[tree.root_id].status == "active"


class TestYAMLBuilder:
    def test_supports_yaml(self):
        builder = YAMLBuilder()
        assert builder.supports("test.yaml") is True
        assert builder.supports("test.yml") is True
        assert builder.supports("test.pdf") is False

    def test_small_openapi_doc_stays_one_chunk_with_params_included(self, sample_openapi_path):
        """Plus d'extraction OpenAPI spécifique (API_ENDPOINT/API_PARAMETER).
        Le fixture est petit (< seuil de taille) : il reste un seul chunk
        générique (type DOCUMENT), résumé ET paramètres inclus — c'est
        justement ce qui corrige le bug historique (paramètres requis
        absents du corpus indexé, voir docstring de structured_data.py).
        Le comportement de fragmentation par endpoint sur un GROS document
        est couvert séparément par test_large_openapi_doc_chunks_per_endpoint."""
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)

        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) == 2  # root + 1 chunk (tout le document, sous le seuil de taille)
        whole = [d for d in docs if d.id != tree.root_id][0]
        assert "symbol" in whole.text
        assert "required" in whole.text.lower()

    def test_large_openapi_doc_chunks_per_endpoint(self, tmp_path):
        """Au-delà du seuil de taille, chaque endpoint devient son propre
        chunk (résumé + paramètres ensemble), sans jamais perdre
        l'information — juste plus fragmenté qu'un petit document."""
        endpoints = "\n".join(
            f"""  /v1/resource{i}:
    post:
      summary: Create resource {i}
      description: {"Lorem ipsum dolor sit amet. " * 20}
      parameters:
        - name: field_{i}
          required: true
          schema:
            type: string
"""
            for i in range(10)
        )
        path = tmp_path / "big_api.yaml"
        path.write_text(f"openapi: '3.0.3'\ninfo:\n  title: X\n  version: '1.0'\npaths:\n{endpoints}", encoding="utf-8")

        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(path), config)
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        # Un chunk par endpoint (10, plus les petits chunks "openapi"/"info"
        # scalaires/plats — non comptés ici, hors sujet du test).
        endpoint_docs = [d for d in docs if "field_" in (d.text or "")]
        assert len(endpoint_docs) == 10
        for d in endpoint_docs:
            assert "required" in d.text.lower()

    def test_no_more_api_endpoint_typed_nodes(self, sample_openapi_path):
        """Non-régression du choix architectural : plus aucun nœud typé
        API_ENDPOINT/API_PARAMETER/API_RESPONSE n'est produit — tout passe
        par le type générique DOCUMENT."""
        config = SourceConfig(source_id="test_api")
        tree = YAMLBuilder().build(str(sample_openapi_path), config)
        assert tree.get_nodes_by_type(NodeType.API_ENDPOINT) == []
        assert tree.get_nodes_by_type(NodeType.API_PARAMETER) == []
        assert tree.get_nodes_by_type(NodeType.API_RESPONSE) == []


class TestMarkdownBuilder:
    def test_supports_md(self):
        builder = MarkdownBuilder()
        assert builder.supports("test.md") is True
        assert builder.supports("test.markdown") is True
        assert builder.supports("test.yaml") is False

    def test_build_markdown(self, sample_markdown_path):
        config = SourceConfig(source_id="test_md")
        builder = MarkdownBuilder()
        tree = builder.build(str(sample_markdown_path), config)

        headings = tree.get_nodes_by_type(NodeType.HEADING)
        assert len(headings) >= 4

        paragraphs = tree.get_nodes_by_type(NodeType.PARAGRAPH)
        assert len(paragraphs) >= 1

        code_blocks = tree.get_nodes_by_type(NodeType.CODE_BLOCK)
        assert len(code_blocks) >= 1

        list_items = tree.get_nodes_by_type(NodeType.LIST_ITEM)
        assert len(list_items) >= 6

    def test_hierarchy_integrity(self, sample_markdown_path):
        config = SourceConfig(source_id="test_md")
        tree = MarkdownBuilder().build(str(sample_markdown_path), config)
        errors = tree.validate_integrity()
        assert len(errors) == 0, f"Integrity errors: {errors}"

    def test_tight_list_items_keep_their_text(self, tmp_path):
        """Régression : markdown-it-py enveloppe le contenu d'un list_item
        dans paragraph_open(hidden)/inline/paragraph_close(hidden) — un
        "inline" ne suit jamais list_item_open directement. L'ancien code
        supposait le contraire et produisait des LIST_ITEM au texte vide
        pour toute liste "tight" (`- item`, sans ligne vide entre éléments),
        le cas le plus courant en markdown."""
        md_path = tmp_path / "tight_list.md"
        md_path.write_text("# Titre\n\n- premier élément\n- second élément\n- troisième élément\n", encoding="utf-8")

        config = SourceConfig(source_id="test_tight_list")
        tree = MarkdownBuilder().build(str(md_path), config)

        list_items = tree.get_nodes_by_type(NodeType.LIST_ITEM)
        assert len(list_items) == 3
        assert [item.text for item in list_items] == ["premier élément", "second élément", "troisième élément"]

    def test_nested_list_items_still_extracted(self, tmp_path):
        """Non-régression : la sous-liste doit rester traitée par la boucle
        externe (pas avalée par le nouveau parsing du list_item parent)."""
        md_path = tmp_path / "nested_list.md"
        md_path.write_text("- item un\n- item deux\n  - item imbriqué\n- item trois\n", encoding="utf-8")

        config = SourceConfig(source_id="test_nested_list")
        tree = MarkdownBuilder().build(str(md_path), config)

        list_items = tree.get_nodes_by_type(NodeType.LIST_ITEM)
        texts = [item.text for item in list_items]
        assert texts == ["item un", "item deux", "item imbriqué", "item trois"]


class TestJSONBuilder:
    def test_supports_json(self):
        builder = JSONBuilder()
        assert builder.supports("test.json") is True
        assert builder.supports("test.yaml") is False

    def test_build_incidents(self, sample_json_path):
        """group_nested n'existe plus — la liste d'enregistrements
        (top-level, ici) est trouvée automatiquement."""
        config = SourceConfig(source_id="test_incidents")
        builder = JSONBuilder()
        tree = builder.build(str(sample_json_path), config)

        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) >= 2

        root = tree.get_node(tree.root_id)
        assert root is not None
        assert root.type == NodeType.DOCUMENT

    def test_integrity(self, sample_json_path):
        config = SourceConfig(source_id="test_incidents")
        tree = JSONBuilder().build(str(sample_json_path), config)
        assert len(tree.validate_integrity()) == 0


class TestStructuredDataBuilderGeneric:
    """StructuredDataBuilder doit ingérer une forme de YAML/JSON jamais vue,
    sans aucune déclaration manuelle de clé — voir discussion Phase 5 sur
    l'extensibilité. Reproduit le bug réel trouvé sur l'ancien manifeste
    alpaca.yaml (group_nested mal renseigné → 0 chunk, silencieusement)
    pour prouver que la découverte automatique le rend impossible par
    construction."""

    def _config(self, source_id="unknown_shape"):
        return SourceConfig(source_id=source_id)

    def test_dict_wrapped_record_list_json(self, tmp_path):
        """Dict racine dont une clé (jamais déclarée nulle part) contient
        une liste de dicts — forme réelle d'alpaca/incidents.json,
        reproduite ici sous un nom de clé arbitraire. La clé "meta" (dict
        de scalaires, sans structure imbriquée) devient aussi son propre
        petit enregistrement plutôt que d'être silencieusement perdue —
        contrairement à l'ancienne approche group_nested qui l'aurait
        ignorée."""
        path = tmp_path / "unknown.json"
        path.write_text(
            '{"meta": {"generated_at": "2026-01-01"}, "widgets": ['
            '{"id": 1, "name": "a", "sub_events": [{"t": 1}, {"t": 2}]},'
            '{"id": 2, "name": "b", "sub_events": [{"t": 3}]}'
            "]}",
            encoding="utf-8",
        )
        tree = JSONBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        # root + "meta" + 2 enregistrements "widgets" ; "sub_events" ne doit
        # pas être fragmenté séparément (reste inclus dans le texte de son
        # enregistrement parent).
        assert len(docs) == 4
        non_root = [d for d in docs if d.id != tree.root_id]
        assert any("sub_events" in (d.text or "") for d in non_root)
        assert any("generated_at" in (d.text or "") for d in non_root)
        assert len(tree.validate_integrity()) == 0

    def test_top_level_record_list_yaml(self, tmp_path):
        """Même logique pour une liste directement à la racine, en YAML
        plutôt que JSON."""
        path = tmp_path / "unknown.yaml"
        path.write_text(
            "- name: alpha\n  value: 1\n- name: beta\n  value: 2\n- name: gamma\n  value: 3\n",
            encoding="utf-8",
        )
        tree = YAMLBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) == 4  # root + 3 enregistrements

    def test_named_record_dict(self, tmp_path):
        """Dict de dicts homogène (ex. forme AsyncAPI-like
        `channels: {name: {...}}`) — pas de liste, mais un enregistrement
        par clé nommée."""
        path = tmp_path / "unknown.json"
        path.write_text(
            '{"channels": {"orderCreated": {"summary": "x"}, "orderCancelled": {"summary": "y"}}}',
            encoding="utf-8",
        )
        tree = JSONBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) == 3  # root + 2 enregistrements nommés

    def test_flat_document_fallback(self, tmp_path):
        """Document entièrement plat (aucune valeur imbriquée) : un seul
        chunk pour tout le document plutôt qu'un chunk par clé scalaire."""
        path = tmp_path / "unknown.json"
        path.write_text('{"key1": "value1", "key2": 42, "key3": true}', encoding="utf-8")
        tree = JSONBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) == 2  # root + 1 chunk unique
        assert len(tree.validate_integrity()) == 0

    def test_mixed_dict_with_one_scalar_key_still_descends_into_others(self, tmp_path):
        """Reproduit la structure OpenAPI (une clé scalaire "openapi" à côté
        de clés structurées "info"/"paths") : la clé scalaire isolée ne doit
        pas empêcher la décomposition des autres clés — chacune est jugée
        indépendamment."""
        path = tmp_path / "api.yaml"
        path.write_text(
            "openapi: '3.0.3'\n"
            "info:\n  title: X\n  version: '1.0'\n"
            "paths:\n"
            "  /v1/orders:\n"
            "    post:\n"
            "      summary: Create order\n"
            "      parameters:\n"
            "        - name: symbol\n          required: true\n"
            "      responses:\n"
            "        '200':\n          description: OK\n",
            encoding="utf-8",
        )
        tree = YAMLBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        non_root = [d for d in docs if d.id != tree.root_id]
        # Un chunk pour "openapi" (scalaire isolé), un pour "info" (dict
        # plat), un pour l'opération POST /v1/orders (dict mixte, sous le
        # seuil de taille -> un seul chunk avec le paramètre inclus).
        post_docs = [d for d in non_root if "symbol" in (d.text or "")]
        assert len(post_docs) == 1
        assert "summary" in post_docs[0].text.lower() or "Create order" in post_docs[0].text


class TestRefResolution:
    """$ref (JSON Pointer interne, RFC 6901) doit être déréférencé avant le
    parcours générique, pour qu'un paramètre partagé référencé par plusieurs
    endpoints apparaisse dans le chunk de chacun plutôt que de nécessiter un
    second chunk séparé pour être compris."""

    def _config(self):
        return SourceConfig(source_id="ref_test")

    def test_referenced_parameter_is_inlined_into_endpoint_chunk(self, tmp_path):
        path = tmp_path / "api.yaml"
        path.write_text(
            "openapi: '3.0.3'\n"
            "components:\n"
            "  parameters:\n"
            "    SymbolParam:\n"
            "      name: symbol\n"
            "      required: true\n"
            "      description: Trading pair symbol\n"
            "paths:\n"
            "  /v1/orders:\n"
            "    get:\n"
            "      summary: List orders\n"
            "      parameters:\n"
            "        - $ref: '#/components/parameters/SymbolParam'\n",
            encoding="utf-8",
        )
        tree = YAMLBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        non_root = [d for d in docs if d.id != tree.root_id]

        endpoint_docs = [d for d in non_root if "List orders" in (d.text or "")]
        assert len(endpoint_docs) == 1
        assert "Trading pair symbol" in endpoint_docs[0].text
        assert "$ref" not in endpoint_docs[0].text

    def test_cycle_leaves_ref_untouched_instead_of_looping(self, tmp_path):
        path = tmp_path / "cyclic.json"
        path.write_text(
            '{"defs": {"a": {"$ref": "#/defs/b"}, "b": {"$ref": "#/defs/a"}}, '
            '"root_field": {"$ref": "#/defs/a"}}',
            encoding="utf-8",
        )
        tree = JSONBuilder().build(str(path), self._config())
        assert len(tree.validate_integrity()) == 0

    def test_unresolvable_ref_left_as_is_without_crashing(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text('{"item": {"$ref": "#/does/not/exist"}}', encoding="utf-8")
        tree = JSONBuilder().build(str(path), self._config())
        docs = tree.get_nodes_by_type(NodeType.DOCUMENT)
        assert any("$ref" in (d.text or "") for d in docs if d.id != tree.root_id)
