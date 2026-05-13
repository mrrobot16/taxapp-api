"""Tests for `app/llm/factory.build_provider_from_env`."""

import pytest

from app.llm import anthropic as anthropic_module
from app.llm import factory as factory_module
from app.llm import gemini as gemini_module
from app.llm import openai as openai_module
from app.llm.base import ProviderError


class _Sentinel:
    """Trivial provider stand-in returned by patched `from_env` classmethods."""

    def __init__(self, label: str) -> None:
        self.label = label


@pytest.fixture(autouse=True)
def _patch_provider_from_env(monkeypatch):
    """Replace each provider's `from_env` with a sentinel-returning stub."""
    monkeypatch.setattr(
        anthropic_module.AnthropicProvider,
        "from_env",
        classmethod(lambda cls: _Sentinel("anthropic")),
    )
    monkeypatch.setattr(
        openai_module.OpenAIProvider,
        "from_env",
        classmethod(lambda cls: _Sentinel("openai")),
    )
    monkeypatch.setattr(
        gemini_module.GeminiProvider,
        "from_env",
        classmethod(lambda cls: _Sentinel("gemini")),
    )


def test_build_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = factory_module.build_provider_from_env()
    assert isinstance(provider, _Sentinel)
    assert provider.label == "anthropic"


def test_build_provider_respects_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert factory_module.build_provider_from_env().label == "anthropic"


def test_build_provider_respects_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert factory_module.build_provider_from_env().label == "openai"


def test_build_provider_respects_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert factory_module.build_provider_from_env().label == "gemini"


def test_build_provider_is_case_insensitive_and_strips(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    assert factory_module.build_provider_from_env().label == "openai"


def test_build_provider_raises_on_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    with pytest.raises(ProviderError, match="Unknown LLM_PROVIDER"):
        factory_module.build_provider_from_env()


def test_build_provider_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    # Empty string is a "no value" -> default. (The factory uses `os.getenv(...)
    # or DEFAULT_PROVIDER`, which treats "" as falsy.)
    assert factory_module.build_provider_from_env().label == "anthropic"
