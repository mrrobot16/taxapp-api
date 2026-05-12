"""Server-Sent Events helpers: payload formatting."""

import json
from collections.abc import AsyncIterator


def format_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_sse(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    """Wrap an async iterator of dict payloads as SSE-formatted strings."""
    async for event in events:
        yield format_sse_event(event)
