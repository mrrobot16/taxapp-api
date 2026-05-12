"""HTTP routes for the Taxapp API."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.rag.pipeline import stream_chat_pipeline
from app.schemas import ChatRequest
from app.web.sse import stream_sse

router = APIRouter()


@router.get("/api/health")
def health(request: Request):
    collection = request.app.state.collection
    if collection is None:
        return {"status": "no_index", "doc_count": 0}
    return {"status": "ok", "doc_count": collection.count()}


@router.post("/api/chat")
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
