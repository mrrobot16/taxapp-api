"""Health check endpoint."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    collection = request.app.state.collection
    if collection is None:
        return {"status": "no_index", "doc_count": 0}
    return {"status": "ok", "doc_count": collection.count()}
