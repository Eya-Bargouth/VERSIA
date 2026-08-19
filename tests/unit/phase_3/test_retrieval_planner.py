import json

from src.llm.interface import BaseLLMClient, LLMConfig, LLMResponse, LLMUsage
from src.retrieval.planner import QueryPlanner


def test_planner_basic():
    qp = QueryPlanner()
    r = qp.plan("What parameter is required for POST /v1/orders?")
    assert r["intent"] in ("factual", "ambiguous")
    assert "/v1/orders" in (r.get("entity") or "")

    r2 = qp.plan("Compare v1 and v2 of the API for currency type")
    assert r2["intent"] == "comparative"


class TestComparativeDetectionBeyondFixedWordList:
    """L'ancienne liste figée (compare/difference/vs/versus/between/différ)
    manquait la formulation française réelle du jeu d'éval Phase 5 ("entre
    la version X et la version Y"), qui ne contient aucun de ces mots —
    0/15 questions de conflit de version étaient routées vers 'comparative'."""

    def test_entre_et_french_version_conflict_phrasing(self):
        qp = QueryPlanner(version_tags=[])
        r = qp.plan(
            "Qu'est-ce qui a changé dans la description de l'endpoint "
            "« GET:/v1/balance/history » entre la version legacy et la "
            "version v2213 de l'API Stripe ?"
        )
        assert r["intent"] == "comparative"

    def test_which_is_better_phrasing(self):
        qp = QueryPlanner(version_tags=[])
        assert qp.plan("Which is better for pagination, cursor or offset?")["intent"] == "comparative"

    def test_distinguish_phrasing(self):
        qp = QueryPlanner(version_tags=[])
        assert qp.plan("Can you distinguish these two authentication flows?")["intent"] == "comparative"

    def test_word_boundary_avoids_substring_false_positive(self):
        """"vs" ne doit matcher qu'en tant que mot, pas comme sous-chaîne
        (ex. à l'intérieur d'un identifiant ou d'un autre mot)."""
        qp = QueryPlanner(version_tags=[])
        r = qp.plan("What does the avset field mean in the response?")
        assert r["intent"] != "comparative"


class TestComparativeVersionTagsFilter:
    """Une question comparative qui nomme 2 versions connues doit filtrer
    Qdrant aux 2, pas les laisser toutes en concurrence dans le top-k (mesure
    Precision@10 : c'était la principale source de faux positifs sur les
    questions version_conflict)."""

    _TAGS = ["legacy", "v2213", "v2293", "v2323"]

    def test_two_named_tags_on_comparative_query_sets_plural_filter(self):
        qp = QueryPlanner(version_tags=self._TAGS)
        r = qp.plan(
            "Qu'est-ce qui a changé dans la description de l'endpoint "
            "entre la version legacy et la version v2213 de l'API Stripe ?"
        )
        assert r["intent"] == "comparative"
        assert r["filters"].get("version_tags") == ["legacy", "v2213"]
        assert "version_tag" not in r["filters"]

    def test_single_named_tag_on_factual_query_stays_singular(self):
        # Évite un chemin contenant "v1" : "v1" + "v2323" (qui contient "v2")
        # déclencherait la garde générique "v1 et v2" (comparaison non
        # spécifique) et désactiverait toute détection de tag ici.
        qp = QueryPlanner(version_tags=self._TAGS)
        r = qp.plan("What parameter is required for GET /accounts in v2323?")
        assert r["filters"].get("version_tag") == "v2323"
        assert "version_tags" not in r["filters"]

    def test_only_one_named_tag_on_comparative_query_stays_singular(self):
        """Une comparaison qui ne nomme qu'une seule version connue ne peut
        pas filtrer aux 2 — repli sur le filtre simple plutôt que de ne rien
        filtrer du tout."""
        qp = QueryPlanner(version_tags=self._TAGS)
        r = qp.plan("Compare the v2213 pagination behaviour to the previous one")
        assert r["intent"] == "comparative"
        assert r["filters"].get("version_tag") == "v2213"
        assert "version_tags" not in r["filters"]

    def test_no_named_tag_on_comparative_query_sets_no_version_filter(self):
        qp = QueryPlanner(version_tags=self._TAGS)
        r = qp.plan("What's the difference between the old and new behaviour of this endpoint?")
        assert r["intent"] == "comparative"
        assert "version_tag" not in r["filters"]
        assert "version_tags" not in r["filters"]

    def test_v1_path_segment_does_not_suppress_real_tag_detection(self):
        """Régression : une question mentionnant un chemin "/v1/..." (quasi
        systématique dans ce corpus) ET un vrai tag "v22xx" (qui contient
        "v2" comme sous-chaîne) déclenchait à tort la garde "v1 et v2
        génériques" et désactivait toute détection de version — 0/15
        questions version_conflict du jeu de test filtraient réellement,
        malgré le mécanisme déjà en place."""
        qp = QueryPlanner(version_tags=self._TAGS)
        r = qp.plan(
            "Qu'est-ce qui a changé dans la description de l'endpoint "
            "« GET:/v1/balance/history » entre la version legacy et la "
            "version v2213 de l'API Stripe ?"
        )
        assert r["filters"].get("version_tags") == ["legacy", "v2213"]


class TestLLMFallback:
    """llm_client était accepté par __init__ mais jamais utilisé dans
    plan() — le fallback promis par le docstring n'existait pas. Vérifie
    qu'il est maintenant réellement appelé, uniquement sur intent
    'ambiguous', et seulement si llm_config est aussi fourni."""

    class _ScriptedLLM(BaseLLMClient):
        def __init__(self, intent: str):
            self.intent = intent
            self.calls = 0

        def complete(self, messages, config):
            self.calls += 1
            return LLMResponse(content=json.dumps({"intent": self.intent}), usage=LLMUsage(), model=config.model)

        async def complete_stream(self, messages, config):
            yield json.dumps({"intent": self.intent})

        def validate_config(self, config):
            return True

    def test_ambiguous_query_is_reclassified_via_llm(self):
        llm = self._ScriptedLLM(intent="factual")
        qp = QueryPlanner(
            llm_client=llm,
            llm_config=LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"),
            version_tags=[],
        )
        result = qp.plan("tell me about the thing")
        assert result["intent"] == "factual"
        assert llm.calls == 1

    def test_llm_not_called_when_rules_already_decided(self):
        llm = self._ScriptedLLM(intent="factual")
        qp = QueryPlanner(
            llm_client=llm,
            llm_config=LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"),
            version_tags=[],
        )
        qp.plan("What is GET /v1/orders?")
        assert llm.calls == 0

    def test_llm_not_called_without_llm_config(self):
        """llm_client seul (sans llm_config) ne doit pas planter — plan()
        reste purement rule-based, comme avant le fix."""
        llm = self._ScriptedLLM(intent="factual")
        qp = QueryPlanner(llm_client=llm, version_tags=[])
        result = qp.plan("tell me about the thing")
        assert result["intent"] == "ambiguous"
        assert llm.calls == 0

    def test_llm_failure_degrades_to_ambiguous_without_crashing(self):
        class _BrokenLLM(BaseLLMClient):
            def complete(self, messages, config):
                raise RuntimeError("juge indisponible")

            async def complete_stream(self, messages, config):
                yield ""

            def validate_config(self, config):
                return True

        qp = QueryPlanner(
            llm_client=_BrokenLLM(),
            llm_config=LLMConfig(provider="ollama", model="qwen2.5:3b-instruct"),
            version_tags=[],
        )
        result = qp.plan("tell me about the thing")
        assert result["intent"] == "ambiguous"
