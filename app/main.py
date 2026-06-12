"""
Taxapp — FastAPI Backend

Exposes the RAG chatbot as a streaming SSE API so any frontend can consume it.

Run with:
    poetry run uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config  # noqa: F401  — import side effects: load `.env` files
from app.api import router
from app.llm import build_provider_from_env
# from app.rag.vectorstore import warmup_collection
from app.utils.logger import install_access_log_middleware, setup_logging

setup_logging()
logger = logging.getLogger("taxapp.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.state.collection = 
    app.state.collection = None
    if app.state.collection is None:
        logger.warning("Knowledge base not indexed yet; /api/chat will return 503.")

    app.state.llm_provider = build_provider_from_env()

    try:
        yield
    finally:
        await app.state.llm_provider.aclose()


app = FastAPI(title="Taxapp API", version="1.0.0", lifespan=lifespan)
install_access_log_middleware(app, logger)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

app.include_router(router)
