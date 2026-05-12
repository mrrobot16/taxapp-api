# Taxapp API

A RAG-powered IRS tax chatbot backend. Answers user tax questions by retrieving
relevant chunks from a local vector index of IRS forms and workflow examples,
then streaming an Anthropic Claude response over Server-Sent Events.

## Architecture

```
User question
     │
     ▼
┌──────────────────────────┐    ┌─────────────────────────┐
│  FastAPI (app/main.py)   │───▶│  app/rag/pipeline.py    │
│  app/api/v0/chat.py      │    │  1. Embed query (Chroma)│
│  POST /api/chat (SSE)    │    │  2. Retrieve top-K      │
                                │  3. Filter by score     │
                                │  4. Build prompt        │
                                │  5. Stream Claude reply │
                                └─────────────────────────┘
                                            │
                                            ▼
                                ┌─────────────────────────┐
                                │  data/chroma_db/        │
                                │  Persistent vector DB   │
                                │  (built by indexer.py)  │
                                └─────────────────────────┘
```

Layout:

```
api/
├── app/                    # Application package
│   ├── main.py             # FastAPI app, CORS, lifespan
│   ├── config.py           # Filesystem paths + `.env` loading
│   ├── constants.py        # Hardcoded literal values (TOP_K, EMBED_MODEL, …)
│   ├── schemas.py          # Pydantic request models
│   ├── prompts.py          # System prompt
│   ├── utils/              # Cross-cutting helpers
│   │   └── logger.py       # Color formatter + access-log middleware
│   ├── api/                # HTTP transport layer
│   │   ├── sse.py          # Server-Sent Events helpers (shared)
│   │   └── v0/             # Version 0 of the API (mounted at /api/)
│   │       ├── chat.py     # POST /api/chat
│   │       └── health.py   # GET  /api/health
│   └── rag/                # Retrieval-Augmented Generation core
│       ├── pipeline.py     # RAG + streaming LLM orchestration
│       ├── retrieval.py
│       ├── vectorstore.py
│       ├── embeddings.py
│       └── llm.py
├── scripts/
│   ├── irs-forms.py        # Downloads IRS form PDFs into data/irs_forms/
│   └── indexer.py          # Chunks + embeds PDFs into Chroma
├── tests/
└── data/                   # Vector DB + raw PDFs (gitignored)
```

The `v0` package name is internal versioning only — it is **not** part of
the URL. Clients continue to hit `/api/chat` and `/api/health`. Future
versions can be added under `app/api/v1/`, etc., and mounted alongside in
`app/api/__init__.py`.

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
     tries each candidate in `app/constants.py` until one succeeds).

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
poetry run uvicorn app.main:app --reload --port 8000
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

Defined in `app/constants.py`:

- `TOP_K = 8` — number of chunks retrieved per query.
- `MAX_HISTORY = 10` — turns of conversation history forwarded to the LLM.
- `MIN_CONTEXT_SCORE = 0.45` — minimum cosine similarity for a chunk to be
  included; if all chunks fall below this, the model returns a "no relevant
  context" fallback instead of guessing.
- `EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"` — embedding model used by both
  the indexer and runtime retrieval (they must match).
- `DEFAULT_ANTHROPIC_MODELS` — ordered fallback list for the chat completion.
