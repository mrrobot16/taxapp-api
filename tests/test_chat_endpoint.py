"""Tests for the `POST /api/chat` endpoint in `app/web/routes.py`.

We replace `routes.stream_chat_pipeline` with a controlled async generator
so the endpoint tests do not depend on Anthropic, Chroma, or the embedding
model.
"""

from app.web import routes
from tests.conftest import FakeCollection, make_app


def _install_pipeline_stub(monkeypatch, events: list[dict], captured: dict):
    """Replace `routes.stream_chat_pipeline` with a stub.

    Records the call args in `captured` and yields the given `events`.
    """

    async def stub(req, collection, client, models, pin_model):
        captured["req"] = req
        captured["collection"] = collection
        captured["client"] = client
        captured["models"] = models
        captured["pin_model"] = pin_model
        for event in events:
            yield event

    monkeypatch.setattr(routes, "stream_chat_pipeline", stub)


async def _read_sse_body(response) -> str:
    chunks: list[str] = []
    async for chunk in response.aiter_text():
        chunks.append(chunk)
    return "".join(chunks)


async def test_chat_503_when_no_collection(client_factory):
    app = make_app(collection=None)
    async with client_factory(app) as client:
        response = await client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Knowledge base not indexed yet."}


async def test_chat_streams_sse_envelope(client_factory, monkeypatch):
    events = [
        {"type": "text", "content": "hi"},
        {"type": "done"},
    ]
    captured: dict = {}
    _install_pipeline_stub(monkeypatch, events, captured)

    app = make_app(
        collection=FakeCollection(count_value=1),
        anthropic_models=["claude-sonnet-4-6"],
        anthropic_client=object(),
    )

    async with client_factory(app) as client:
        async with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["connection"] == "keep-alive"
            assert response.headers["x-accel-buffering"] == "no"
            body = await _read_sse_body(response)

    expected = (
        'data: {"type": "text", "content": "hi"}\n\n'
        'data: {"type": "done"}\n\n'
    )
    assert body == expected


async def test_chat_passes_request_to_pipeline(client_factory, monkeypatch):
    captured: dict = {}
    _install_pipeline_stub(monkeypatch, [{"type": "done"}], captured)

    app = make_app(
        collection=FakeCollection(),
        anthropic_models=["m1", "m2"],
        anthropic_client="sentinel-client",
    )

    payload = {
        "message": "what is form 1040?",
        "history": [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier reply"},
        ],
        "top_k": 3,
    }

    async with client_factory(app) as client:
        async with client.stream("POST", "/api/chat", json=payload) as response:
            await _read_sse_body(response)

    req = captured["req"]
    assert req.message == "what is form 1040?"
    assert [(m.role, m.content) for m in req.history] == [
        ("user", "earlier"),
        ("assistant", "earlier reply"),
    ]
    assert req.top_k == 3
    assert captured["collection"] is app.state.collection
    assert captured["client"] == "sentinel-client"
    assert captured["models"] is app.state.anthropic_models


async def test_chat_request_applies_top_k_default(client_factory, monkeypatch):
    """Omitting `top_k` should fall back to the default from `config.TOP_K`."""
    from app.config import TOP_K

    captured: dict = {}
    _install_pipeline_stub(monkeypatch, [{"type": "done"}], captured)

    app = make_app(collection=FakeCollection(), anthropic_models=["m1"])
    async with client_factory(app) as client:
        async with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
            await _read_sse_body(response)

    assert captured["req"].top_k == TOP_K
    assert captured["req"].history == []


async def test_chat_pin_model_hook_mutates_state(client_factory, monkeypatch):
    captured: dict = {}
    _install_pipeline_stub(monkeypatch, [{"type": "done"}], captured)

    app = make_app(
        collection=FakeCollection(),
        anthropic_models=["m1", "m2"],
    )

    async with client_factory(app) as client:
        async with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
            await _read_sse_body(response)

    pin = captured["pin_model"]
    assert callable(pin)
    pin("m2")
    assert app.state.anthropic_models == ["m2"]


async def test_chat_invalid_body_returns_422(client_factory):
    app = make_app(collection=FakeCollection(), anthropic_models=["m1"])
    async with client_factory(app) as client:
        response = await client.post("/api/chat", json={})

    assert response.status_code == 422


async def test_chat_rejects_get(client_factory):
    app = make_app(collection=FakeCollection(), anthropic_models=["m1"])
    async with client_factory(app) as client:
        response = await client.get("/api/chat")

    assert response.status_code == 405
