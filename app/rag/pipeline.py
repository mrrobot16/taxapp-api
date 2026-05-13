"""Streaming RAG chat pipeline that runs natively on the asyncio loop."""

import asyncio
import logging
import textwrap
from collections.abc import AsyncIterator

from app.constants import MAX_HISTORY
from app.llm.base import LLMProvider, ProviderError
from app.prompts import SYSTEM_PROMPT
from app.rag.retrieval import build_context_block, filter_relevant_chunks, retrieve_context
from app.schemas import ChatRequest

logger = logging.getLogger("taxapp.api")

async def stream_chat_pipeline(
    chat_request: ChatRequest,
    collection,
    provider: LLMProvider,
) -> AsyncIterator[dict]:
    """Run the RAG + LLM streaming flow and yield SSE-shaped dict events.

    The single off-loop hop is the synchronous Chroma + embedding call; the
    LLM stream is consumed natively on the event loop so per-token overhead
    is negligible.
    """
    yield {"type": "phase", "label": "Searching IRS knowledge base"}
    retrieved_chunks = await asyncio.to_thread(
        retrieve_context, collection, chat_request.message, chat_request.top_k
    )

    chunks = filter_relevant_chunks(retrieved_chunks)
    if not chunks:
        yield {"type": "phase", "label": "No relevant IRS sources found"}
        yield {
            "type": "text",
            "content": (
                "I don't have enough relevant IRS context to answer this confidently. "
                "Try rephrasing the question with the exact form, publication, "
                "or tax topic you need."
            ),
        }
        yield {"type": "sources", "sources": []}
        yield {"type": "done"}
        return

    context_block = build_context_block(chunks)

    user_content = textwrap.dedent(f"""
        ## Retrieved IRS Knowledge Base Context

        {context_block}

        ---

        ## Question

        {chat_request.message}
    """).strip()

    history_payload = [
        {"role": message.role, "content": message.content} for message in chat_request.history[-MAX_HISTORY * 2:]
    ]
    messages_payload = history_payload + [{"role": "user", "content": user_content}]

    yield {"type": "phase", "label": "Preparing your answer"}
    try:
        async for text_chunk in provider.stream(
            system=SYSTEM_PROMPT,
            messages=messages_payload,
            max_tokens=2048,
        ):
            yield {"type": "text", "content": text_chunk}
    except ProviderError as exception:
        yield {"type": "error", "message": str(exception)}
        return
    except Exception as exception:
        yield {"type": "error", "message": str(exception)}
        return

    yield {"type": "phase", "label": "Finalizing sources"}
    sources = [
        {
            "text": chunk["text"][:600],
            "metadata": chunk["metadata"],
            "score": round(chunk["score"], 3),
        }
        for chunk in chunks
    ]
    yield {"type": "sources", "sources": sources}
    yield {"type": "done"}
