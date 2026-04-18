"""
Taxapp — FastAPI Backend

Exposes the RAG chatbot as a streaming SSE API so any frontend can consume it.

Run with:
    poetry run uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import os
import queue
import textwrap
import threading
from pathlib import Path

import chromadb
import torch
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = SCRIPT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "tax_knowledge"
# EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"

TOP_K = 8
MAX_HISTORY = 10
MIN_CONTEXT_SCORE = 0.45
DEFAULT_ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-latest",
]

SYSTEM_PROMPT = """You are an expert US tax CPA assistant ("Taxapp") with deep knowledge \
of 2025 IRS forms, instructions, publications, and tax law. You only answer tax-related questions.

Rules:
- Base every answer strictly on the retrieved IRS context provided in the user turn.
- If the context doesn't contain enough information to answer confidently, say so clearly.
- Always mention the specific IRS form numbers or publication numbers that are relevant.
- Do not invent facts, citations, or form numbers.
- Keep a professional, helpful tone.

Formatting:
- Use ## and ### headings to break answers into clear, scannable sections.
- Use **bold** for IRS form numbers, publication numbers, and key terms on first mention.
- Use numbered lists (1. 2. 3.) for step-by-step instructions or sequential processes.
- Use bullet lists for non-sequential items, requirements, or options.
- Use markdown tables when comparing forms, thresholds, filing requirements, or deadlines.
- Use --- horizontal rules to separate distinct topics within a single answer.
- Keep paragraphs concise — 2-3 sentences maximum per paragraph.
- Start the response directly with content; do not begin with a heading that restates the question."""


load_dotenv(REPO_ROOT / ".env")
load_dotenv(SCRIPT_DIR / ".env")


app = FastAPI(title="Taxapp API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbeddingFunction:
    """Same wrapper used by the indexer — must match exactly."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=_get_device())

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, show_progress_bar=False).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def name(self) -> str:
        return EMBED_MODEL


_collection = None


def get_candidate_models() -> list[str]:
    configured = os.getenv("ANTHROPIC_MODEL", "").strip()
    models = [configured] + DEFAULT_ANTHROPIC_MODELS if configured else DEFAULT_ANTHROPIC_MODELS[:]
    seen = set()
    deduped = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def is_model_access_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "forbidden" in msg
        or "request not allowed" in msg
        or "not found" in msg
        or "does not exist" in msg
        or ("model" in msg and "access" in msg)
    )


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    if not CHROMA_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = LocalEmbeddingFunction(EMBED_MODEL)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        return _collection
    except Exception as e:
        print(f"[api] Could not load collection: {e}")
        return None


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


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source_label = {
            "form_summary": f"IRS Form Summary — {meta.get('form', 'unknown')}",
            "irs_form": f"IRS Form — {meta.get('form', 'unknown')}",
            "form_instructions": f"Form Instructions — {meta.get('form', 'unknown')}",
            "form_instructions_combined": f"Form & Instructions — {meta.get('form', 'unknown')}",
            "irs_publication": f"IRS Publication — {meta.get('publication', 'unknown')}",
            "prompt_example": f"Tax Scenario Example — {meta.get('scenario', '')}",
            "flow_example": f"Tax Workflow — {meta.get('flow', '')}",
        }.get(meta.get("source", ""), meta.get("file", ""))
        parts.append(f"[Source {i}: {source_label}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def format_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = []
    top_k: int = TOP_K


@app.get("/api/health")
def health():
    collection = get_collection()
    if collection is None:
        return {"status": "no_index", "doc_count": 0}
    return {"status": "ok", "doc_count": collection.count()}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    collection = get_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="Knowledge base not indexed yet.")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured on the server.")

    def run_pipeline(event_queue: queue.Queue):
        def emit(payload: dict):
            event_queue.put(payload)

        try:
            emit({"type": "phase", "label": "Searching IRS knowledge base"})
            retrieved_chunks = retrieve_context(collection, req.message, top_k=req.top_k)

            chunks = filter_relevant_chunks(retrieved_chunks)
            if not chunks:
                emit({"type": "phase", "label": "No relevant IRS sources found"})
                emit(
                    {
                        "type": "text",
                        "content": (
                            "I don't have enough relevant IRS context to answer this confidently. "
                            "Try rephrasing the question with the exact form, publication, "
                            "or tax topic you need."
                        ),
                    }
                )
                emit({"type": "sources", "sources": []})
                emit({"type": "done"})
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

            emit({"type": "phase", "label": "Preparing your answer"})
            client = Anthropic(api_key=anthropic_api_key)
            streamed = False
            last_error = None

            for model in get_candidate_models():
                try:
                    with client.messages.stream(
                        model=model,
                        max_tokens=2048,
                        system=SYSTEM_PROMPT,
                        messages=messages_payload,
                    ) as stream:
                        streamed = True
                        for text_chunk in stream.text_stream:
                            emit({"type": "text", "content": text_chunk})
                    break
                except Exception as e:
                    last_error = e
                    if is_model_access_error(e):
                        continue
                    emit({"type": "error", "message": str(e)})
                    return

            if not streamed:
                msg = (
                    "No allowed Anthropic model found for this API key. "
                    "Set ANTHROPIC_MODEL in .env to a model your key can access."
                )
                if last_error is not None:
                    msg = f"{msg} ({last_error})"
                emit({"type": "error", "message": msg})
                return

            emit({"type": "phase", "label": "Finalizing sources"})
            sources = [
                {
                    "text": c["text"][:600],
                    "metadata": c["metadata"],
                    "score": round(c["score"], 3),
                }
                for c in chunks
            ]
            emit({"type": "sources", "sources": sources})
            emit({"type": "done"})
        finally:
            event_queue.put(None)

    async def generate():
        event_queue: queue.Queue[dict | None] = queue.Queue()
        threading.Thread(target=run_pipeline, args=(event_queue,), daemon=True).start()
        while True:
            event = await asyncio.to_thread(event_queue.get)
            if event is None:
                break
            yield format_sse_event(event)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
