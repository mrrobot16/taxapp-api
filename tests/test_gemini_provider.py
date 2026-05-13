"""Tests for `app/llm/gemini.py`.

The google-genai SDK is faked at the
`client.aio.models.generate_content_stream(...)` boundary so these tests
never touch the network.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.base import ProviderError
from app.llm.gemini import DEFAULT_GEMINI_MODEL, GeminiProvider, _to_gemini_contents


@dataclass
class _Chunk:
    text: str | None


class _FakeChunkStream:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "_FakeChunkStream":
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeAioModels:
    def __init__(self, behaviors: list[Any]) -> None:
        self._behaviors = behaviors
        self.calls: list[dict] = []

    async def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        if not self._behaviors:
            return _FakeChunkStream([])
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        return _FakeChunkStream(behavior)


class _FakeAio:
    def __init__(self, models: _FakeAioModels) -> None:
        self.models = models


class FakeGenAIClient:
    """Stand-in for `google.genai.Client` covering only what we use."""

    def __init__(self, behaviors: list[Any] | None = None) -> None:
        self._models = _FakeAioModels(behaviors or [])
        self.aio = _FakeAio(self._models)

    @property
    def calls(self) -> list[dict]:
        return self._models.calls


def _chunks(*texts: str | None) -> list[_Chunk]:
    return [_Chunk(text=t) for t in texts]


async def _drain(agen) -> list[str]:
    return [chunk async for chunk in agen]


# ---------------------------------------------------------------------------
# _to_gemini_contents
# ---------------------------------------------------------------------------


def test_to_gemini_contents_maps_assistant_role_to_model():
    contents = _to_gemini_contents(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow up"},
        ]
    )
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "follow up"}]},
    ]


def test_to_gemini_contents_drops_system_messages():
    contents = _to_gemini_contents(
        [
            {"role": "system", "content": "leak"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_from_env_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider.from_env()


def test_from_env_uses_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider = GeminiProvider.from_env()
    assert provider.name == "gemini"
    assert provider.model == DEFAULT_GEMINI_MODEL


def test_from_env_respects_model_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert GeminiProvider.from_env().model == "gemini-2.5-pro"


def test_from_env_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    provider = GeminiProvider.from_env()
    assert provider.model == DEFAULT_GEMINI_MODEL


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


async def test_stream_yields_only_chunks_with_text():
    client = FakeGenAIClient(behaviors=[_chunks("Hello", None, "", " world")])
    provider = GeminiProvider(client, "gemini-2.5-flash")

    out = await _drain(
        provider.stream(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert out == ["Hello", " world"]


async def test_stream_passes_system_via_config_and_translates_messages():
    client = FakeGenAIClient(behaviors=[_chunks("ok")])
    provider = GeminiProvider(client, "gemini-2.5-flash")

    await _drain(
        provider.stream(
            system="sys-prompt",
            messages=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "earlier reply"},
                {"role": "user", "content": "now"},
            ],
            max_tokens=128,
        )
    )

    call = client.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "earlier"}]},
        {"role": "model", "parts": [{"text": "earlier reply"}]},
        {"role": "user", "parts": [{"text": "now"}]},
    ]
    config = call["config"]
    assert config.system_instruction == "sys-prompt"
    assert config.max_output_tokens == 128


async def test_stream_propagates_sdk_exceptions():
    client = FakeGenAIClient(behaviors=[RuntimeError("quota exceeded")])
    provider = GeminiProvider(client, "gemini-2.5-flash")

    with pytest.raises(RuntimeError, match="quota exceeded"):
        await _drain(provider.stream(system="s", messages=[], max_tokens=8))


async def test_aclose_is_noop():
    provider = GeminiProvider(FakeGenAIClient(), "gemini-2.5-flash")
    assert await provider.aclose() is None
