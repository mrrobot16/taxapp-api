"""Tests for `app/llm/anthropic.py`.

Covers the model-resolution helpers, the access-error classifier, and the
end-to-end behavior of `AnthropicProvider.stream()` including fallback and
pinning. The Anthropic SDK is faked at the `client.messages.stream(...)`
boundary so these tests never touch the network.
"""

from collections.abc import Iterable
from typing import Any

import anthropic
import httpx
import pytest

from app.constants import DEFAULT_ANTHROPIC_MODELS
from app.llm.anthropic import (
    AnthropicProvider,
    is_anthropic_access_error,
    resolve_anthropic_models,
)
from app.llm.base import ProviderError


# ---------------------------------------------------------------------------
# Fakes for the Anthropic SDK surface that AnthropicProvider depends on.
# ---------------------------------------------------------------------------


class _FakeTextStream:
    def __init__(self, chunks: Iterable[str]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "_FakeTextStream":
        return self

    async def __anext__(self) -> str:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeStreamCM:
    def __init__(self, chunks: Iterable[str]) -> None:
        self.text_stream = _FakeTextStream(chunks)

    async def __aenter__(self) -> "_FakeStreamCM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeMessages:
    """Mimics `AsyncAnthropic.messages` for our provider tests.

    `behaviors` is consumed in order, once per `stream()` call:
      - `list[str]` -> text chunks to yield from the streaming response
      - `BaseException` -> raised synchronously from `stream()` (matches the
        SDK, which raises on context-manager entry for auth/model errors).
    """

    def __init__(self, behaviors: list[Any] | None = None) -> None:
        self._behaviors: list[Any] = list(behaviors) if behaviors is not None else [[]]
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if not self._behaviors:
            return _FakeStreamCM([])
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        return _FakeStreamCM(behavior)


class FakeAnthropic:
    """Stand-in for `anthropic.AsyncAnthropic` covering only what we use."""

    def __init__(self, behaviors: list[Any] | None = None) -> None:
        self.messages = FakeMessages(behaviors)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def make_anthropic_error(error_cls: type, status_code: int = 404) -> Exception:
    """Construct a real anthropic.* status error usable for tests."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return error_cls(message="test error", response=response, body=None)


async def _drain(agen) -> list[str]:
    return [chunk async for chunk in agen]


# ---------------------------------------------------------------------------
# resolve_anthropic_models
# ---------------------------------------------------------------------------


def test_resolve_anthropic_models_blank_env_returns_defaults(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert resolve_anthropic_models() == DEFAULT_ANTHROPIC_MODELS


def test_resolve_anthropic_models_empty_string_returns_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    assert resolve_anthropic_models() == DEFAULT_ANTHROPIC_MODELS


def test_resolve_anthropic_models_prepends_custom_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "custom-model")
    result = resolve_anthropic_models()
    assert result[0] == "custom-model"
    for default in DEFAULT_ANTHROPIC_MODELS:
        assert default in result


def test_resolve_anthropic_models_dedupes_when_env_matches_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODELS[0])
    result = resolve_anthropic_models()
    assert result.count(DEFAULT_ANTHROPIC_MODELS[0]) == 1
    assert result[0] == DEFAULT_ANTHROPIC_MODELS[0]
    assert len(result) == len(DEFAULT_ANTHROPIC_MODELS)


def test_resolve_anthropic_models_strips_whitespace(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "  custom-model  ")
    assert resolve_anthropic_models()[0] == "custom-model"


# ---------------------------------------------------------------------------
# is_anthropic_access_error
# ---------------------------------------------------------------------------


def test_is_anthropic_access_error_true_for_not_found():
    err = make_anthropic_error(anthropic.NotFoundError, status_code=404)
    assert is_anthropic_access_error(err) is True


def test_is_anthropic_access_error_true_for_permission_denied():
    err = make_anthropic_error(anthropic.PermissionDeniedError, status_code=403)
    assert is_anthropic_access_error(err) is True


def test_is_anthropic_access_error_false_for_other_exceptions():
    assert is_anthropic_access_error(RuntimeError("boom")) is False
    assert is_anthropic_access_error(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# AnthropicProvider.from_env
# ---------------------------------------------------------------------------


def test_from_env_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider.from_env()


def test_from_env_constructs_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    provider = AnthropicProvider.from_env()
    assert provider.name == "anthropic"
    assert provider.models == DEFAULT_ANTHROPIC_MODELS


# ---------------------------------------------------------------------------
# AnthropicProvider.stream
# ---------------------------------------------------------------------------


async def test_stream_happy_path_yields_chunks_and_calls_first_model():
    client = FakeAnthropic(behaviors=[["Hello ", "world"]])
    provider = AnthropicProvider(client, ["m1", "m2"])

    chunks = await _drain(
        provider.stream(system="sys", messages=[{"role": "user", "content": "hi"}], max_tokens=128)
    )

    assert chunks == ["Hello ", "world"]
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == "m1"
    assert call["max_tokens"] == 128
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    # First model already in front -> no pin reorder.
    assert provider.models == ["m1", "m2"]


@pytest.mark.parametrize(
    "error_cls",
    [anthropic.NotFoundError, anthropic.PermissionDeniedError],
)
async def test_stream_falls_back_to_next_model_on_access_error(error_cls):
    client = FakeAnthropic(
        behaviors=[
            make_anthropic_error(error_cls, status_code=403),
            ["only chunk"],
        ]
    )
    provider = AnthropicProvider(client, ["m1", "m2"])

    chunks = await _drain(
        provider.stream(system="sys", messages=[{"role": "user", "content": "hi"}], max_tokens=128)
    )

    assert chunks == ["only chunk"]
    assert [c["model"] for c in client.messages.calls] == ["m1", "m2"]
    # The winning model has been pinned to the front.
    assert provider.models == ["m2", "m1"]


async def test_stream_pin_persists_across_calls():
    client = FakeAnthropic(
        behaviors=[
            make_anthropic_error(anthropic.NotFoundError),
            ["first"],
            ["second"],
        ]
    )
    provider = AnthropicProvider(client, ["m1", "m2"])

    await _drain(provider.stream(system="s", messages=[], max_tokens=8))
    await _drain(provider.stream(system="s", messages=[], max_tokens=8))

    # After pinning, the second call hits m2 directly without re-trying m1.
    assert [c["model"] for c in client.messages.calls] == ["m1", "m2", "m2"]


async def test_stream_propagates_non_access_exception_without_fallback():
    client = FakeAnthropic(behaviors=[RuntimeError("boom"), ["never used"]])
    provider = AnthropicProvider(client, ["m1", "m2"])

    with pytest.raises(RuntimeError, match="boom"):
        await _drain(provider.stream(system="s", messages=[], max_tokens=8))

    # Second model should NOT have been tried.
    assert [c["model"] for c in client.messages.calls] == ["m1"]


async def test_stream_raises_provider_error_when_all_models_are_access_denied():
    client = FakeAnthropic(
        behaviors=[
            make_anthropic_error(anthropic.NotFoundError),
            make_anthropic_error(anthropic.PermissionDeniedError),
        ]
    )
    provider = AnthropicProvider(client, ["m1", "m2"])

    with pytest.raises(ProviderError, match="No allowed Anthropic model"):
        await _drain(provider.stream(system="s", messages=[], max_tokens=8))


async def test_stream_raises_provider_error_with_empty_model_list():
    client = FakeAnthropic()
    provider = AnthropicProvider(client, [])

    with pytest.raises(ProviderError, match="No Anthropic models configured"):
        await _drain(provider.stream(system="s", messages=[], max_tokens=8))

    assert client.messages.calls == []


async def test_aclose_closes_underlying_client():
    client = FakeAnthropic()
    provider = AnthropicProvider(client, ["m1"])

    await provider.aclose()

    assert client.closed is True
