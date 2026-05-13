"""Streaming chat endpoint."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.rag.pipeline import stream_chat_pipeline
from app.schemas import ChatRequest
from app.utils.sse import stream_sse

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    state = request.app.state
    if state.collection is None:
        raise HTTPException(status_code=503, detail="Knowledge base not indexed yet.")

    events = stream_chat_pipeline(chat_request, state.collection, state.llm_provider)
    return StreamingResponse(
        stream_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
