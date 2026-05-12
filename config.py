"""Configuration constants and `.env` loading for the Taxapp API.

Importing this module triggers `load_dotenv` for both the repo-root and
api-local `.env` files, so any module that depends on environment variables
should ensure `config` is imported before reading `os.getenv`.
"""

from pathlib import Path

from dotenv import load_dotenv

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

load_dotenv(REPO_ROOT / ".env")
load_dotenv(SCRIPT_DIR / ".env")
