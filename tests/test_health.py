"""Tests for the `GET /api/health` endpoint in [api/routes.py](api/routes.py)."""

from tests.conftest import FakeCollection, make_app


async def test_health_no_index_when_collection_missing(client_factory):
    app = make_app(collection=None)
    async with client_factory(app) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "no_index", "doc_count": 0}


async def test_health_ok_with_collection(client_factory):
    app = make_app(collection=FakeCollection(count_value=42))
    async with client_factory(app) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "doc_count": 42}


async def test_health_rejects_post(client_factory):
    app = make_app(collection=None)
    async with client_factory(app) as client:
        response = await client.post("/api/health")

    assert response.status_code == 405
