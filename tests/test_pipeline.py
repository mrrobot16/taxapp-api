"""Unit tests for `stream_chat_pipeline` in [api/pipeline.py](api/pipeline.py)."""

import anthropic
import pytest

from config import MAX_HISTORY, MIN_CONTEXT_SCORE
from pipeline import stream_chat_pipeline
from schemas import ChatRequest, HistoryMessage
from tests.conftest import (
    FakeAnthropic,
    FakeCollection,
    collect_events,
    make_anthropic_error,
)


def _low_score_collection() -> FakeCollection:
    """A collection whose chunks are all below MIN_CONTEXT_SCORE."""
    # score = 1 - distance; with distance=0.9 score=0.1 < MIN_CONTEXT_SCORE (0.45)
    return FakeCollection(
        query_result={
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [[{"source": "irs_form", "form": "1040"}, {"source": "irs_form", "form": "W2"}]],
            "distances": [[0.9, 0.95]],
        }
    )


def _relevant_collection() -> FakeCollection:
    """A collection whose chunks easily clear MIN_CONTEXT_SCORE."""
    long_doc = "x" * 800
    return FakeCollection(
        query_result={
            "documents": [["short doc", long_doc]],
            "metadatas": [
                [
                    {"source": "irs_form", "form": "1040"},
                    {"source": "irs_publication", "publication": "17"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
    )


def _basic_request(**overrides) -> ChatRequest:
    payload = {"message": "What is form 1040?", "history": [], "top_k": 4}
    payload.update(overrides)
    return ChatRequest(**payload)


async def test_pipeline_emits_no_context_fallback_when_all_chunks_filtered():
    collection = _low_score_collection()
    client = FakeAnthropic()

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=["m1"],
            pin_model=None,
        )
    )

    types = [e["type"] for e in events]
    assert types == ["phase", "phase", "text", "sources", "done"]
    assert events[0] == {"type": "phase", "label": "Searching IRS knowledge base"}
    assert events[1] == {"type": "phase", "label": "No relevant IRS sources found"}
    assert "I don't have enough relevant IRS context" in events[2]["content"]
    assert events[3] == {"type": "sources", "sources": []}
    assert events[4] == {"type": "done"}

    # Anthropic must never be called when no relevant chunks exist.
    assert client.messages.calls == []


async def test_pipeline_happy_path_streams_text_and_sources():
    collection = _relevant_collection()
    client = FakeAnthropic(behaviors=[["Hello ", "world"]])
    pinned: list[str] = []

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=["m1"],
            pin_model=pinned.append,
        )
    )

    types = [e["type"] for e in events]
    assert types == [
        "phase",
        "phase",
        "text",
        "text",
        "phase",
        "sources",
        "done",
    ]
    assert events[0]["label"] == "Searching IRS knowledge base"
    assert events[1]["label"] == "Preparing your answer"
    assert events[2] == {"type": "text", "content": "Hello "}
    assert events[3] == {"type": "text", "content": "world"}
    assert events[4]["label"] == "Finalizing sources"

    sources = events[5]["sources"]
    assert len(sources) == 2
    assert all(len(s["text"]) <= 600 for s in sources)
    # score must be rounded to 3 decimals
    for s in sources:
        rounded = round(s["score"], 3)
        assert s["score"] == rounded
    assert events[6] == {"type": "done"}

    # First model succeeded — pin_model should NOT have been called.
    assert pinned == []
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["model"] == "m1"


@pytest.mark.parametrize(
    "error_cls",
    [anthropic.NotFoundError, anthropic.PermissionDeniedError],
)
async def test_pipeline_falls_back_to_next_model_on_access_error(error_cls):
    collection = _relevant_collection()
    client = FakeAnthropic(
        behaviors=[
            make_anthropic_error(error_cls, status_code=403),
            ["only chunk"],
        ]
    )
    pinned: list[str] = []

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=["m1", "m2"],
            pin_model=pinned.append,
        )
    )

    types = [e["type"] for e in events]
    assert "error" not in types
    assert types[-1] == "done"

    # Both models were attempted.
    assert [c["model"] for c in client.messages.calls] == ["m1", "m2"]
    # Because the chosen model differs from models[0], pin_model is called.
    assert pinned == ["m2"]


async def test_pipeline_yields_error_event_on_non_access_exception():
    collection = _relevant_collection()
    boom = RuntimeError("kaboom")
    client = FakeAnthropic(behaviors=[boom])

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=["m1", "m2"],
            pin_model=None,
        )
    )

    assert events[-1] == {"type": "error", "message": "kaboom"}
    # Pipeline stops after the error: no "done", no "sources".
    assert not any(e["type"] == "done" for e in events)
    assert not any(e["type"] == "sources" for e in events)
    # Second model should not have been tried.
    assert [c["model"] for c in client.messages.calls] == ["m1"]


async def test_pipeline_yields_error_when_no_models_available():
    collection = _relevant_collection()
    client = FakeAnthropic()

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=[],
            pin_model=None,
        )
    )

    assert events[-1]["type"] == "error"
    assert "No allowed Anthropic model" in events[-1]["message"]
    assert client.messages.calls == []


async def test_pipeline_truncates_history_to_max_history_pairs():
    collection = _relevant_collection()
    client = FakeAnthropic(behaviors=[["ok"]])

    # 2 * MAX_HISTORY pairs of turns = 4 * MAX_HISTORY messages total. Far
    # over the cap so the trimming is observable.
    too_many = [
        HistoryMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
        for i in range(4 * MAX_HISTORY)
    ]
    req = ChatRequest(message="latest", history=too_many, top_k=2)

    await collect_events(
        stream_chat_pipeline(
            req,
            collection,
            client,
            models=["m1"],
            pin_model=None,
        )
    )

    call = client.messages.calls[0]
    sent_messages = call["messages"]
    # Last message must be the synthesized user turn with the question.
    assert sent_messages[-1]["role"] == "user"
    assert "latest" in sent_messages[-1]["content"]
    history_sent = sent_messages[:-1]
    assert len(history_sent) == 2 * MAX_HISTORY
    # And they must be the *most recent* slice.
    expected_first = f"msg {4 * MAX_HISTORY - 2 * MAX_HISTORY}"
    assert history_sent[0]["content"] == expected_first


async def test_pipeline_passes_top_k_to_retrieval():
    """`req.top_k` should reach `collection.query(..., n_results=top_k)`."""
    collection = _relevant_collection()
    client = FakeAnthropic(behaviors=[["ok"]])
    req = _basic_request(top_k=3)

    await collect_events(
        stream_chat_pipeline(req, collection, client, models=["m1"], pin_model=None)
    )

    assert collection.query_calls[0]["n_results"] == 3
    assert collection.query_calls[0]["query_texts"] == [req.message]


async def test_pipeline_clips_sources_text_to_600_chars():
    """Verify the 600-char cap on `sources[*].text` from pipeline.py."""
    huge = "y" * 1000
    collection = FakeCollection(
        query_result={
            "documents": [[huge]],
            "metadatas": [[{"source": "irs_form", "form": "1040"}]],
            "distances": [[0.1]],
        }
    )
    client = FakeAnthropic(behaviors=[["ok"]])

    events = await collect_events(
        stream_chat_pipeline(
            _basic_request(),
            collection,
            client,
            models=["m1"],
            pin_model=None,
        )
    )

    sources_events = [e for e in events if e["type"] == "sources"]
    assert sources_events
    assert len(sources_events[0]["sources"][0]["text"]) == 600


# Sanity check: MIN_CONTEXT_SCORE assumptions used by the helpers above.
def test_min_context_score_assumption():
    assert 0.0 <= MIN_CONTEXT_SCORE <= 1.0
    assert MIN_CONTEXT_SCORE > 0.05  # the "low score" fixture must be below it
