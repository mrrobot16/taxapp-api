"""Tests for the SSE helpers in `app/utils/sse.py`."""

import json

from app.utils.sse import format_sse_event, stream_sse


def test_format_sse_event_serializes_dict():
    out = format_sse_event({"type": "text", "content": "hi"})
    assert out.endswith("\n\n")
    assert out.startswith("data: ")
    payload = out[len("data: "):-2]
    assert json.loads(payload) == {"type": "text", "content": "hi"}


def test_format_sse_event_handles_nested_values():
    out = format_sse_event({"type": "sources", "sources": [{"text": "a", "score": 0.5}]})
    payload = out[len("data: "):-2]
    assert json.loads(payload) == {
        "type": "sources",
        "sources": [{"text": "a", "score": 0.5}],
    }


async def test_stream_sse_wraps_each_event_in_sse_frame():
    async def gen():
        yield {"type": "phase", "label": "step-1"}
        yield {"type": "text", "content": "hello"}
        yield {"type": "done"}

    frames = [frame async for frame in stream_sse(gen())]

    assert frames == [
        'data: {"type": "phase", "label": "step-1"}\n\n',
        'data: {"type": "text", "content": "hello"}\n\n',
        'data: {"type": "done"}\n\n',
    ]


async def test_stream_sse_empty_iterator_yields_nothing():
    async def gen():
        if False:
            yield  # pragma: no cover - keeps this a generator

    frames = [frame async for frame in stream_sse(gen())]
    assert frames == []
