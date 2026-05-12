"""Hardcoded constants for the Taxapp API.

Literal, environment-agnostic values that don't change at runtime.
Filesystem paths and `.env` loading live in `app.config` instead.
"""

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
