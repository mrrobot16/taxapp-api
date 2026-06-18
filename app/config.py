"""Filesystem paths and `.env` loading for the Taxapp API.

Importing this module triggers `load_dotenv` for both the repo-root and
api-local `.env` files, so any module that depends on environment variables
should ensure `app.config` is imported before reading `os.getenv`.

Hardcoded literal constants live in `app.constants` instead.
"""

from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
API_DIR = APP_DIR.parent
REPO_ROOT = API_DIR.parent

DATA_DIR = API_DIR / "data"
SCRIPTS_DIR = API_DIR / "scripts"
CHROMA_DIR = DATA_DIR / "chroma_db"
IRS_FORMS_DIR = DATA_DIR / "irs_forms"
# FLOWS_DIR = DATA_DIR / "flows"
FLOWS_DIR = SCRIPTS_DIR / "flows"

load_dotenv(REPO_ROOT / ".env")
load_dotenv(API_DIR / ".env")
