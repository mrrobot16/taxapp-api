# Taxapp API

A RAG-powered IRS tax chatbot backend. Answers user tax questions by retrieving
relevant chunks from a local vector index of IRS forms and workflow examples,
then streaming an Anthropic Claude response over Server-Sent Events.

## Architecture

```
User question
     │
     ▼
┌──────────────────────┐    ┌────────────────────────┐
│  FastAPI (api.py)    │───▶│  pipeline.py            │
│  POST /api/chat (SSE)│    │  1. Embed query (Chroma)│
└──────────────────────┘    │  2. Retrieve top-K      │
                            │  3. Filter by score     │
                            │  4. Build prompt        │
                            │  5. Stream Claude reply │
                            └────────────────────────┘
                                       │
                                       ▼
                            ┌────────────────────────┐
                            │  data/chroma_db/        │
                            │  Persistent vector DB   │
                            │  (built by indexer.py)  │
                            └────────────────────────┘
```

Key modules:

- `api.py` — FastAPI app, CORS, lifespan (warms up Chroma + Anthropic client).
- `routes.py` — `/api/health` and `/api/chat` endpoints.
- `pipeline.py` — RAG + streaming LLM pipeline, yields SSE-shaped events.
- `retrieval.py` — Chroma query, score filtering, context block formatting.
- `vectorstore.py` — Persistent Chroma collection warmup.
- `embeddings.py` — SentenceTransformer embedding wrapper (`multi-qa-MiniLM-L6-cos-v1`).
- `llm.py` — Anthropic model resolution with fallback across `DEFAULT_ANTHROPIC_MODELS`.
- `prompts.py` — System prompt.
- `config.py` — Constants and `.env` loading.
- `scripts/irs-forms.py` — Downloads IRS form PDFs from `irs.gov` into `data/irs_forms/`.
- `scripts/indexer.py` — Chunks PDFs + flow examples and embeds them into Chroma.

## Prerequisites

- Python `>=3.11,<3.15`
- [Poetry](https://python-poetry.org/)
- An Anthropic API key

## Setup

1. Install dependencies:

   ```bash
   poetry install
   ```

2. Create `.env` from the example and fill in your keys:

   ```bash
   cp .env_example .env
   ```

   Required:

   - `ANTHROPIC_API_KEY` — Anthropic API key (must have access to one of the
     models in `DEFAULT_ANTHROPIC_MODELS`, or set `ANTHROPIC_MODEL` to override).

   Optional:

   - `ANTHROPIC_MODEL` — Pin a specific Claude model (otherwise the server
     tries each candidate in `config.py` until one succeeds).

3. Download the IRS form PDFs (run once):

   ```bash
   poetry run python scripts/irs-forms.py
   ```

   This populates `data/irs_forms/` with ~thousands of IRS PDFs from
   `https://www.irs.gov/pub/irs-pdf/`.

4. Build the vector index (run once, or any time `data/` changes):

   ```bash
   poetry run python scripts/indexer.py
   ```

   Use `--reset` to delete and rebuild the collection from scratch.
   The indexer auto-selects `mps` / `cuda` / `cpu` for embedding.

## Run the API

```bash
poetry run uvicorn api:app --reload --port 8000
```

Endpoints:

- `GET  /api/health` — returns `{"status": "ok", "doc_count": N}` once the index
  is loaded, or `{"status": "no_index", "doc_count": 0}` if not.
- `POST /api/chat` — streams the assistant reply as SSE.

### Request shape

```json
{
  "message": "How do I report 1099-NEC income?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "top_k": 8
}
```

### SSE event types

The stream emits JSON objects under `data:` lines:

- `{"type": "phase",   "label": "..."}` — UI progress hint
- `{"type": "text",    "content": "..."}` — token chunk to append
- `{"type": "sources", "sources": [...]}` — retrieved chunks + scores
- `{"type": "error",   "message": "..."}`
- `{"type": "done"}` — terminal event

### Quick curl test

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a W-2?"}'
```

## Tests

```bash
poetry run pytest
```

`pytest.ini` enables asyncio auto mode and points at the `tests/` directory.

## Configuration knobs

Defined in `config.py`:

- `TOP_K = 8` — number of chunks retrieved per query.
- `MAX_HISTORY = 10` — turns of conversation history forwarded to the LLM.
- `MIN_CONTEXT_SCORE = 0.45` — minimum cosine similarity for a chunk to be
  included; if all chunks fall below this, the model returns a "no relevant
  context" fallback instead of guessing.
- `EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"` — embedding model used by both
  the indexer and runtime retrieval (they must match).
- `DEFAULT_ANTHROPIC_MODELS` — ordered fallback list for the chat completion.
