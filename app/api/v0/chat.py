"""Streaming chat endpoint."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.utils.sse import stream_sse
from app.rag.pipeline import stream_chat_pipeline
from app.schemas import ChatRequest

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    state = request.app.state
    collection = state.collection
    if collection is None:
        raise HTTPException(status_code=503, detail="Knowledge base not indexed yet.")

    def pin_model(model: str) -> None:
        state.anthropic_models = [model]

    events = stream_chat_pipeline(
        req,
        collection,
        state.anthropic_client,
        state.anthropic_models,
        pin_model,
    )
    return StreamingResponse(
        stream_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
