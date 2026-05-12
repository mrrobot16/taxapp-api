"""Shared fixtures and fakes for the Taxapp API test suite.

The real app in `app/main.py` wires up `app.state` from a lifespan that
requires `ANTHROPIC_API_KEY`, a Chroma collection, and loads a
SentenceTransformer model. We skip all of that here: each test gets a fresh
`FastAPI()` with the router mounted and only the `app.state` attributes it
actually needs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.api import router


def make_app(
    *,
    collection: Any = None,
    anthropic_models: list[str] | None = None,
    anthropic_client: Any = None,
) -> FastAPI:
    """Build a bare FastAPI app with only the state the routes need."""
    app = FastAPI()
    app.state.collection = collection
    app.state.anthropic_models = anthropic_models or []
    app.state.anthropic_client = anthropic_client
    app.include_router(router)
    return app


@pytest_asyncio.fixture
async def client_factory():
    """Yields an async factory that builds an httpx client bound to an app.

    Tests typically do:
        async with client_factory(app) as client:
            ...
    """

    clients: list[httpx.AsyncClient] = []

    def _factory(app: FastAPI) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        clients.append(client)
        return client

    yield _factory

    for c in clients:
        await c.aclose()


class FakeCollection:
    """Minimal Chroma collection stand-in.

    `query_result` must use the shape that `app/rag/retrieval.py` expects:
    `{"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}`.
    """

    def __init__(
        self,
        *,
        count_value: int = 0,
        query_result: dict | None = None,
    ) -> None:
        self._count = count_value
        self._query_result = query_result or {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.query_calls: list[dict] = []

    def count(self) -> int:
        return self._count

    def query(self, query_texts, n_results, include):
        self.query_calls.append(
            {"query_texts": query_texts, "n_results": n_results, "include": include}
        )
        return self._query_result


class _FakeTextStream:
    """Async iterator over a configured list of text chunks."""

    def __init__(self, chunks: Iterable[str]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "_FakeTextStream":
        return self

    async def __anext__(self) -> str:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeStreamCM:
    """Async context manager mimicking `client.messages.stream(...)`."""

    def __init__(self, chunks: Iterable[str]) -> None:
        self.text_stream = _FakeTextStream(chunks)

    async def __aenter__(self) -> "_FakeStreamCM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeMessages:
    """Mimics `AsyncAnthropic.messages` for our pipeline tests.

    `behaviors` is consumed in order, once per `stream()` call:
      - a list[str] -> text chunks to yield from the streaming response
      - an Exception instance -> raised synchronously from `stream()`
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


def make_anthropic_error(error_cls: type, status_code: int = 404) -> Exception:
    """Construct a real anthropic.* status error usable for tests."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return error_cls(message="test error", response=response, body=None)


async def collect_events(agen: AsyncIterator[dict]) -> list[dict]:
    """Drain an async iterator into a list (helper for pipeline tests)."""
    return [event async for event in agen]


@pytest.fixture
def fake_collection_factory():
    """Factory fixture so tests can build collections with custom data."""
    return FakeCollection


@pytest.fixture
def fake_anthropic_factory():
    return FakeAnthropic
