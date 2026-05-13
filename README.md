# Taxapp API

A RAG-powered IRS tax chatbot backend. Answers user tax questions by retrieving
relevant chunks from a local vector index of IRS forms and workflow examples,
then streaming an LLM response over Server-Sent Events. The LLM is pluggable —
Anthropic Claude, OpenAI, or Google Gemini, chosen by a single env var.

## Architecture

```
User question
     │
     ▼
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
│  FastAPI (app/main.py)   │───▶│  app/rag/pipeline.py     │───▶│  app/llm/        │
│  app/api/v0/chat.py      │    │  1. Embed query (Chroma) │    │  LLMProvider     │
│  POST /api/chat (SSE)    │    │  2. Retrieve top-K       │    │  ├── Anthropic   │
                                │  3. Filter by score      │    │  ├── OpenAI      │
                                │  4. Build prompt         │    │  └── Gemini      │
                                │  5. Stream LLM reply     │    └──────────────────┘
                                └──────────────────────────┘
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
│   │   ├── logger.py       # Color formatter + access-log middleware
│   │   └── sse.py          # Server-Sent Events helpers
│   ├── api/                # HTTP transport layer
│   │   └── v0/             # Version 0 of the API (mounted at /api/)
│   │       ├── chat.py     # POST /api/chat
│   │       └── health.py   # GET  /api/health
│   ├── llm/                # Pluggable LLM providers (see "Choosing the LLM")
│   │   ├── base.py         # LLMProvider Protocol + ProviderError
│   │   ├── anthropic.py    # Claude (model fallback + pinning)
│   │   ├── openai.py       # GPT-4o / GPT-4o-mini / …
│   │   ├── gemini.py       # Gemini 2.5 Flash / Pro / …
│   │   └── factory.py      # build_provider_from_env()
│   └── rag/                # Retrieval-Augmented Generation core
│       ├── pipeline.py     # RAG orchestration (provider-agnostic)
│       ├── retrieval.py
│       ├── vectorstore.py
│       └── embeddings.py
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
- An API key from at least one supported LLM vendor (Anthropic, OpenAI, or Google)

## Setup

1. Install dependencies:

   ```bash
   poetry install
   ```

2. Create `.env` from the example and fill in your keys:

   ```bash
   cp .env_example .env
   ```

   You only need credentials for the provider you actually select via
   `LLM_PROVIDER`. See [Choosing the LLM](#choosing-the-llm) below for the full
   list of env vars per provider.

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

## Choosing the LLM

The chat endpoint is **provider-agnostic**. The RAG pipeline talks to an
`LLMProvider` Protocol (`app/llm/base.py`); the concrete implementation is
selected at startup by `build_provider_from_env()` based on the `LLM_PROVIDER`
environment variable. Swapping models is a `.env` change + restart — no code
edits required.

### Switching providers (env-only)

Set two or three variables in `.env`:

| Provider  | `LLM_PROVIDER` | API key env var                       | Model env var (optional) | Default model        |
|-----------|----------------|---------------------------------------|--------------------------|----------------------|
| Anthropic | `anthropic`    | `ANTHROPIC_API_KEY`                   | `ANTHROPIC_MODEL`        | first of `DEFAULT_ANTHROPIC_MODELS` (with fallback) |
| OpenAI    | `openai`       | `OPENAI_API_KEY`                      | `OPENAI_MODEL`           | `gpt-4o-mini`        |
| Gemini    | `gemini`       | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `GEMINI_MODEL`           | `gemini-2.5-flash`   |

If `LLM_PROVIDER` is unset, the server defaults to `anthropic`. You do **not**
need to supply keys or install SDKs for providers you aren't using — each
implementation is imported lazily by `app/llm/factory.py`, so missing optional
deps for unselected providers won't break startup.

### Examples

Claude (default, with a pinned model):

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

GPT-4o:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

Gemini 2.5 Pro:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
```

After editing `.env`, **restart uvicorn** — the provider is constructed once in
the FastAPI lifespan ([`app/main.py`](app/main.py)) and cached on
`app.state.llm_provider`.

### How Anthropic model fallback works

If `ANTHROPIC_MODEL` is set, it is prepended to `DEFAULT_ANTHROPIC_MODELS` in
[`app/constants.py`](app/constants.py). The provider tries each model in order
until one is accepted by your API key, then **pins the winner** for the
remainder of the process so subsequent requests skip dead candidates. This is
Anthropic-specific behavior; the OpenAI and Gemini providers use a single
configured model.

### Adding a new provider

Drop a new file under `app/llm/` that implements the `LLMProvider` Protocol
defined in [`app/llm/base.py`](app/llm/base.py):

```python
class LLMProvider(Protocol):
    name: str
    def stream(self, *, system: str, messages: list[dict], max_tokens: int) -> AsyncIterator[str]: ...
    async def aclose(self) -> None: ...
```

Then add a branch for it in `build_provider_from_env()`
([`app/llm/factory.py`](app/llm/factory.py)). Nothing in `app/rag/pipeline.py`
or `app/api/` needs to change.

## Tests

```bash
poetry run pytest
```

`pytest.ini` enables asyncio auto mode and points at the `tests/` directory.

## Configuration knobs

Defined in [`app/constants.py`](app/constants.py):

- `TOP_K = 8` — number of chunks retrieved per query.
- `MAX_HISTORY = 10` — turns of conversation history forwarded to the LLM.
- `MIN_CONTEXT_SCORE = 0.45` — minimum cosine similarity for a chunk to be
  included; if all chunks fall below this, the model returns a "no relevant
  context" fallback instead of guessing.
- `EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"` — embedding model used by both
  the indexer and runtime retrieval (they must match).
- `DEFAULT_ANTHROPIC_MODELS` — ordered fallback list used by `AnthropicProvider`.

Per-provider defaults (override via env, see [Choosing the LLM](#choosing-the-llm)):

- `DEFAULT_OPENAI_MODEL = "gpt-4o-mini"` — in [`app/llm/openai.py`](app/llm/openai.py).
- `DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"` — in [`app/llm/gemini.py`](app/llm/gemini.py).
