"""Tests prompt_templates — ordre question/contexte, injection du diff."""

import pytest

from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_message

pytestmark = pytest.mark.phase4


class TestBuildUserMessage:
    def test_question_comes_before_context(self):
        chunks = [{"chunk_id": "abc", "text": "Some text", "payload": {}}]
        msg = build_user_message("What is X?", chunks)
        assert msg.index("What is X?") < msg.index("Some text")

    def test_chunk_id_marker_present_for_citation_grounding(self):
        chunks = [{"chunk_id": "chunk-42", "text": "Relevant content", "payload": {}}]
        msg = build_user_message("Q", chunks)
        assert "chunk-42" in msg

    def test_no_chunks_says_so_explicitly(self):
        msg = build_user_message("Q", [])
        assert "Aucun contexte" in msg

    def test_diff_explanation_appended_when_present(self):
        msg = build_user_message("Q", [], diff_explanation="POST:/v1/orders modifié")
        assert "Différences détectées" in msg
        assert "POST:/v1/orders modifié" in msg

    def test_diff_explanation_absent_when_none(self):
        msg = build_user_message("Q", [])
        assert "Différences détectées" not in msg


class TestSystemPrompt:
    def test_specifies_zero_to_one_range_for_scores(self):
        assert "0.0" in SYSTEM_PROMPT and "1.0" in SYSTEM_PROMPT

    def test_forbids_out_of_context_knowledge(self):
        assert "contexte" in SYSTEM_PROMPT.lower()
