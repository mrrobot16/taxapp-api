"""Pydantic request schemas for the Taxapp API."""

from pydantic import BaseModel

from app.config import TOP_K


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = []
    top_k: int = TOP_K
