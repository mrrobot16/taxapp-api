"""Tests for the `GET /api/health` endpoint in `app/api/v0/health.py`."""

from datetime import datetime

from app.constants import COLLECTION_NAME, EMBED_MODEL, LLM_PROVIDER
from tests.conftest import FakeCollection, make_app


async def test_health_no_index_when_collection_missing(client_factory):
    app = make_app(collection=None)
    async with client_factory(app) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    timestamp = payload.pop("timestamp")
    datetime.fromisoformat(timestamp)
    assert payload == {
        "status": "no_index",
        "doc_count": 0,
        "llm_provider": LLM_PROVIDER,
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
    }


async def test_health_ok_with_collection(client_factory):
    app = make_app(collection=FakeCollection(count_value=42))
    async with client_factory(app) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    timestamp = payload.pop("timestamp")
    datetime.fromisoformat(timestamp)
    assert payload == {
        "status": "ok",
        "doc_count": 42,
        "llm_provider": LLM_PROVIDER,
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
    }


async def test_health_rejects_post(client_factory):
    app = make_app(collection=None)
    async with client_factory(app) as client:
        response = await client.post("/api/health")

    assert response.status_code == 405
