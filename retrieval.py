"""RAG retrieval and context-building helpers."""

from config import MIN_CONTEXT_SCORE, TOP_K

_SOURCE_LABEL_TEMPLATES: dict[str, str] = {
    "form_summary": "IRS Form Summary — {form}",
    "irs_form": "IRS Form — {form}",
    "form_instructions": "Form Instructions — {form}",
    "form_instructions_combined": "Form & Instructions — {form}",
    "irs_publication": "IRS Publication — {publication}",
    "prompt_example": "Tax Scenario Example — {scenario}",
    "flow_example": "Tax Workflow — {flow}",
}


def retrieve_context(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "score": 1 - dist})
    return chunks


def filter_relevant_chunks(chunks: list[dict], min_score: float = MIN_CONTEXT_SCORE) -> list[dict]:
    return [chunk for chunk in chunks if chunk["score"] >= min_score]


def _source_label(meta: dict) -> str:
    template = _SOURCE_LABEL_TEMPLATES.get(meta.get("source", ""))
    if template is None:
        return meta.get("file", "")
    return template.format(
        form=meta.get("form", "unknown"),
        publication=meta.get("publication", "unknown"),
        scenario=meta.get("scenario", ""),
        flow=meta.get("flow", ""),
    )


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {_source_label(chunk['metadata'])}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
