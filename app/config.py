"""Configuration constants and `.env` loading for the Taxapp API.

Importing this module triggers `load_dotenv` for both the repo-root and
api-local `.env` files, so any module that depends on environment variables
should ensure `app.config` is imported before reading `os.getenv`.
"""

from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
API_DIR = APP_DIR.parent
REPO_ROOT = API_DIR.parent

DATA_DIR = API_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
IRS_FORMS_DIR = DATA_DIR / "irs_forms"
FLOWS_DIR = DATA_DIR / "flows"

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
load_dotenv(API_DIR / ".env")
