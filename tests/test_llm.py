"""Tests for the helpers in `app/rag/llm.py`."""

import anthropic

from app.config import DEFAULT_ANTHROPIC_MODELS
from app.rag.llm import get_candidate_models, is_model_access_error
from tests.conftest import make_anthropic_error


def test_get_candidate_models_blank_env_returns_defaults(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert get_candidate_models() == DEFAULT_ANTHROPIC_MODELS


def test_get_candidate_models_empty_string_returns_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    assert get_candidate_models() == DEFAULT_ANTHROPIC_MODELS


def test_get_candidate_models_prepends_custom_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "custom-model")
    result = get_candidate_models()
    assert result[0] == "custom-model"
    # All defaults must still be present, after the custom one.
    for default in DEFAULT_ANTHROPIC_MODELS:
        assert default in result


def test_get_candidate_models_dedupes_when_env_matches_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODELS[0])
    result = get_candidate_models()
    assert result.count(DEFAULT_ANTHROPIC_MODELS[0]) == 1
    assert result[0] == DEFAULT_ANTHROPIC_MODELS[0]
    assert len(result) == len(DEFAULT_ANTHROPIC_MODELS)


def test_get_candidate_models_strips_whitespace(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "  custom-model  ")
    assert get_candidate_models()[0] == "custom-model"


def test_is_model_access_error_true_for_not_found():
    err = make_anthropic_error(anthropic.NotFoundError, status_code=404)
    assert is_model_access_error(err) is True


def test_is_model_access_error_true_for_permission_denied():
    err = make_anthropic_error(anthropic.PermissionDeniedError, status_code=403)
    assert is_model_access_error(err) is True


def test_is_model_access_error_false_for_other_exceptions():
    assert is_model_access_error(RuntimeError("boom")) is False
    assert is_model_access_error(ValueError("nope")) is False
