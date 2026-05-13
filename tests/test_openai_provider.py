"""Tests for `app/llm/openai.py`.

The OpenAI SDK is faked at the `client.chat.completions.create(...)`
boundary so these tests never touch the network.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.base import ProviderError
from app.llm.openai import DEFAULT_OPENAI_MODEL, OpenAIProvider


@dataclass
class _Delta:
    content: str | None


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Chunk:
    choices: list[_Choice]


class _FakeChunkStream:
    """Async iterator over a configured list of chunk objects."""

    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "_FakeChunkStream":
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


class _FakeCompletions:
    def __init__(self, behaviors: list[Any]) -> None:
        self._behaviors = behaviors
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._behaviors:
            return _FakeChunkStream([])
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        return _FakeChunkStream(behavior)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeAsyncOpenAI:
    """Stand-in for `openai.AsyncOpenAI` covering only what we use."""

    def __init__(self, behaviors: list[Any] | None = None) -> None:
        self._completions = _FakeCompletions(behaviors or [])
        self.chat = _FakeChat(self._completions)
        self.closed = False

    @property
    def calls(self) -> list[dict]:
        return self._completions.calls

    async def close(self) -> None:
        self.closed = True


def _chunks(*texts: str | None) -> list[_Chunk]:
    return [_Chunk(choices=[_Choice(delta=_Delta(content=t))]) for t in texts]


async def _drain(agen) -> list[str]:
    return [chunk async for chunk in agen]


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_from_env_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider.from_env()


def test_from_env_uses_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    provider = OpenAIProvider.from_env()
    assert provider.name == "openai"
    assert provider.model == DEFAULT_OPENAI_MODEL


def test_from_env_respects_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert OpenAIProvider.from_env().model == "gpt-4o"


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


async def test_stream_yields_only_non_empty_content_chunks():
    client = FakeAsyncOpenAI(behaviors=[_chunks("Hello, ", None, "", "world")])
    provider = OpenAIProvider(client, "gpt-4o-mini")

    out = await _drain(
        provider.stream(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert out == ["Hello, ", "world"]


async def test_stream_prepends_system_message_and_forwards_args():
    client = FakeAsyncOpenAI(behaviors=[_chunks("ok")])
    provider = OpenAIProvider(client, "gpt-4o-mini")

    await _drain(
        provider.stream(
            system="sys-prompt",
            messages=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "earlier reply"},
                {"role": "user", "content": "now"},
            ],
            max_tokens=256,
        )
    )

    call = client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["max_tokens"] == 256
    assert call["stream"] is True
    assert call["messages"][0] == {"role": "system", "content": "sys-prompt"}
    assert call["messages"][1:] == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
        {"role": "user", "content": "now"},
    ]


async def test_stream_skips_chunks_with_no_choices():
    chunk_no_choices = _Chunk(choices=[])
    client = FakeAsyncOpenAI(behaviors=[[chunk_no_choices, *_chunks("hi")]])
    provider = OpenAIProvider(client, "gpt-4o-mini")

    out = await _drain(provider.stream(system="s", messages=[], max_tokens=8))

    assert out == ["hi"]


async def test_stream_propagates_sdk_exceptions():
    client = FakeAsyncOpenAI(behaviors=[RuntimeError("network down")])
    provider = OpenAIProvider(client, "gpt-4o-mini")

    with pytest.raises(RuntimeError, match="network down"):
        await _drain(provider.stream(system="s", messages=[], max_tokens=8))


async def test_aclose_closes_underlying_client():
    client = FakeAsyncOpenAI()
    provider = OpenAIProvider(client, "gpt-4o-mini")

    await provider.aclose()

    assert client.closed is True
