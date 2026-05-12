"""Streaming RAG chat pipeline that runs natively on the asyncio loop."""

import asyncio
import logging
import textwrap
from collections.abc import AsyncIterator, Callable

from anthropic import AsyncAnthropic

from app.config import MAX_HISTORY
from app.logging_utils import Colors
from app.prompts import SYSTEM_PROMPT
from app.rag.llm import is_model_access_error
from app.rag.retrieval import build_context_block, filter_relevant_chunks, retrieve_context
from app.schemas import ChatRequest

logger = logging.getLogger("taxapp.api")


async def stream_chat_pipeline(
    req: ChatRequest,
    collection,
    client: AsyncAnthropic,
    models: list[str],
    pin_model: Callable[[str], None] | None = None,
) -> AsyncIterator[dict]:
    """Run the RAG + LLM streaming flow and yield SSE-shaped dict events.

    The single off-loop hop is the synchronous Chroma + embedding call; the
    LLM stream is consumed natively on the event loop so per-token overhead
    is negligible.
    """
    yield {"type": "phase", "label": "Searching IRS knowledge base"}
    retrieved_chunks = await asyncio.to_thread(
        retrieve_context, collection, req.message, req.top_k
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

        {req.message}
    """).strip()

    history_payload = [
        {"role": m.role, "content": m.content} for m in req.history[-MAX_HISTORY * 2:]
    ]
    messages_payload = history_payload + [{"role": "user", "content": user_content}]

    yield {"type": "phase", "label": "Preparing your answer"}
    streamed = False
    last_error: Exception | None = None
    chosen_model: str | None = None

    for model in models:
        try:
            logger.info("Trying model: %s%s%s", Colors.MAGENTA, model, Colors.RESET)
            async with client.messages.stream(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages_payload,
            ) as stream:
                streamed = True
                chosen_model = model
                async for text_chunk in stream.text_stream:
                    yield {"type": "text", "content": text_chunk}
            break
        except Exception as e:
            last_error = e
            if not streamed and is_model_access_error(e):
                continue
            yield {"type": "error", "message": str(e)}
            return

    if not streamed:
        msg = (
            "No allowed Anthropic model found for this API key. "
            "Set ANTHROPIC_MODEL in .env to a model your key can access."
        )
        if last_error is not None:
            msg = f"{msg} ({last_error})"
        yield {"type": "error", "message": msg}
        return

    if pin_model is not None and chosen_model is not None and models[0] != chosen_model:
        pin_model(chosen_model)

    yield {"type": "phase", "label": "Finalizing sources"}
    sources = [
        {
            "text": c["text"][:600],
            "metadata": c["metadata"],
            "score": round(c["score"], 3),
        }
        for c in chunks
    ]
    yield {"type": "sources", "sources": sources}
    yield {"type": "done"}
