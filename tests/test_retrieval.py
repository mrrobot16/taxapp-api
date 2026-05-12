"""Tests for the pure helpers in `app/rag/retrieval.py`."""

import pytest

from app.constants import MIN_CONTEXT_SCORE
from app.rag.retrieval import (
    build_context_block,
    filter_relevant_chunks,
    retrieve_context,
)
from tests.conftest import FakeCollection


def test_filter_relevant_chunks_drops_below_threshold():
    chunks = [
        {"text": "keep", "metadata": {}, "score": MIN_CONTEXT_SCORE},
        {"text": "drop", "metadata": {}, "score": MIN_CONTEXT_SCORE - 0.01},
        {"text": "also keep", "metadata": {}, "score": 0.99},
    ]
    result = filter_relevant_chunks(chunks)
    assert [c["text"] for c in result] == ["keep", "also keep"]


def test_filter_relevant_chunks_explicit_min_score():
    chunks = [
        {"text": "a", "metadata": {}, "score": 0.5},
        {"text": "b", "metadata": {}, "score": 0.8},
    ]
    assert [c["text"] for c in filter_relevant_chunks(chunks, min_score=0.7)] == ["b"]


def test_build_context_block_formats_each_known_source():
    chunks = [
        {"text": "A", "metadata": {"source": "form_summary", "form": "1040"}, "score": 0.9},
        {"text": "B", "metadata": {"source": "irs_form", "form": "W2"}, "score": 0.9},
        {"text": "C", "metadata": {"source": "form_instructions", "form": "1040"}, "score": 0.9},
        {
            "text": "D",
            "metadata": {"source": "form_instructions_combined", "form": "1040"},
            "score": 0.9,
        },
        {"text": "E", "metadata": {"source": "irs_publication", "publication": "17"}, "score": 0.9},
        {"text": "F", "metadata": {"source": "prompt_example", "scenario": "AMT"}, "score": 0.9},
        {"text": "G", "metadata": {"source": "flow_example", "flow": "self-employed"}, "score": 0.9},
    ]

    block = build_context_block(chunks)

    assert "[Source 1: IRS Form Summary \u2014 1040]" in block
    assert "[Source 2: IRS Form \u2014 W2]" in block
    assert "[Source 3: Form Instructions \u2014 1040]" in block
    assert "[Source 4: Form & Instructions \u2014 1040]" in block
    assert "[Source 5: IRS Publication \u2014 17]" in block
    assert "[Source 6: Tax Scenario Example \u2014 AMT]" in block
    assert "[Source 7: Tax Workflow \u2014 self-employed]" in block
    # Sources are separated by the canonical horizontal rule.
    assert "\n\n---\n\n" in block


def test_build_context_block_falls_back_to_file_for_unknown_source():
    chunks = [
        {"text": "X", "metadata": {"source": "something_else", "file": "foo.pdf"}, "score": 0.9},
        {"text": "Y", "metadata": {}, "score": 0.9},  # no source, no file -> ""
    ]
    block = build_context_block(chunks)
    assert "[Source 1: foo.pdf]" in block
    assert "[Source 2: ]" in block


def test_retrieve_context_zips_results_and_converts_distance_to_score():
    collection = FakeCollection(
        query_result={
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [
                [
                    {"source": "irs_form", "form": "1040"},
                    {"source": "irs_form", "form": "W2"},
                ]
            ],
            "distances": [[0.1, 0.4]],
        }
    )

    result = retrieve_context(collection, query="hello", top_k=5)

    assert len(result) == 2
    assert result[0]["text"] == "doc-a"
    assert result[0]["metadata"] == {"source": "irs_form", "form": "1040"}
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[1]["text"] == "doc-b"
    assert result[1]["metadata"] == {"source": "irs_form", "form": "W2"}
    assert result[1]["score"] == pytest.approx(0.6)

    assert collection.query_calls[0]["query_texts"] == ["hello"]
    assert collection.query_calls[0]["n_results"] == 5
    assert collection.query_calls[0]["include"] == ["documents", "metadatas", "distances"]
