"""Tests unitaires — retry/backoff sur erreurs transitoires
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_baseline_generation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_baseline_generation", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_baseline_generation"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rbg():
    return _load_module()


class TestRetryOnTransientErrors:
    def test_succeeds_immediately_without_retry(self, rbg, monkeypatch):
        calls = []
        monkeypatch.setattr(rbg, "run_one_question", lambda components, q: calls.append(1) or {"ok": True})
        monkeypatch.setattr(rbg.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("ne doit pas attendre si succès direct")))
        result = rbg.run_one_question_with_retry({}, {"question": "Q?"})
        assert result == {"ok": True}
        assert len(calls) == 1

    def test_retries_once_then_succeeds(self, rbg, monkeypatch):
        attempts = {"n": 0}

        def flaky(components, q):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("Ollama request failed: timed out")
            return {"ok": True}

        sleep_calls = []
        monkeypatch.setattr(rbg, "run_one_question", flaky)
        monkeypatch.setattr(rbg.time, "sleep", lambda s: sleep_calls.append(s))

        result = rbg.run_one_question_with_retry({}, {"question": "Q?"})
        assert result == {"ok": True}
        assert attempts["n"] == 2
        assert sleep_calls == [rbg.RETRY_BACKOFF_SECONDS]

    def test_raises_after_exhausting_retries(self, rbg, monkeypatch):
        def always_fails(components, q):
            raise RuntimeError("échec de validation JSON du juge")

        monkeypatch.setattr(rbg, "run_one_question", always_fails)
        monkeypatch.setattr(rbg.time, "sleep", lambda s: None)

        with pytest.raises(RuntimeError, match="échec de validation JSON"):
            rbg.run_one_question_with_retry({}, {"question": "Q?"})

    def test_attempts_bounded_by_retry_max_attempts(self, rbg, monkeypatch):
        attempts = {"n": 0}

        def always_fails(components, q):
            attempts["n"] += 1
            raise RuntimeError("timeout")

        monkeypatch.setattr(rbg, "run_one_question", always_fails)
        monkeypatch.setattr(rbg.time, "sleep", lambda s: None)

        with pytest.raises(RuntimeError):
            rbg.run_one_question_with_retry({}, {"question": "Q?"})
        assert attempts["n"] == rbg.RETRY_MAX_ATTEMPTS
