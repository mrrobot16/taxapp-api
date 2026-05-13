"""Shared fixtures and fakes for the Taxapp API test suite.

The real app in `app/main.py` wires up `app.state` from a lifespan that
requires API keys, a Chroma collection, and a SentenceTransformer model. We
skip all of that here: each test gets a fresh `FastAPI()` with the router
mounted and only the `app.state` attributes it actually needs.
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
    llm_provider: Any = None,
) -> FastAPI:
    """Build a bare FastAPI app with only the state the routes need."""
    app = FastAPI()
    app.state.collection = collection
    app.state.llm_provider = llm_provider
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


class FakeProvider:
    """Provider-agnostic stand-in for `LLMProvider`.

    `behaviors` is consumed once per `stream()` call:
      - `list[str]` -> text chunks to yield
      - `BaseException` instance -> raised on first iteration
    """

    name = "fake"

    def __init__(self, behaviors: list[Any] | None = None) -> None:
        self._behaviors: list[Any] = list(behaviors) if behaviors is not None else [[]]
        self.calls: list[dict] = []
        self.closed: bool = False

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self.calls.append(
            {"system": system, "messages": messages, "max_tokens": max_tokens}
        )
        if not self._behaviors:
            return
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        for chunk in behavior:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def collect_events(agen: AsyncIterator[dict]) -> list[dict]:
    """Drain an async iterator into a list (helper for pipeline tests)."""
    return [event async for event in agen]


@pytest.fixture
def fake_collection_factory():
    """Factory fixture so tests can build collections with custom data."""
    return FakeCollection


@pytest.fixture
def fake_provider_factory():
    return FakeProvider


@pytest.fixture
def fake_relevant_chunks() -> Iterable[dict]:
    """Convenience: 2 chunks safely above MIN_CONTEXT_SCORE."""
    return [
        {"text": "doc-a", "metadata": {"source": "irs_form", "form": "1040"}, "score": 0.9},
        {"text": "doc-b", "metadata": {"source": "irs_form", "form": "W2"}, "score": 0.8},
    ]
