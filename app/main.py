"""
Taxapp — FastAPI Backend

Exposes the RAG chatbot as a streaming SSE API so any frontend can consume it.

Run with:
    poetry run uvicorn app.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config  # noqa: F401  — import side effects: load `.env` files
from app.logging_utils import install_access_log_middleware, setup_logging
from app.rag.llm import get_candidate_models
from app.rag.vectorstore import warmup_collection
from app.web.routes import router

setup_logging()
logger = logging.getLogger("taxapp.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured on the server.")

    app.state.collection = warmup_collection()
    if app.state.collection is None:
        logger.warning("Knowledge base not indexed yet; /api/chat will return 503.")

    app.state.anthropic_models = get_candidate_models()
    app.state.anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)

    try:
        yield
    finally:
        await app.state.anthropic_client.close()


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
