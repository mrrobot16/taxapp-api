"""Health check endpoint."""

from datetime import datetime 
from fastapi import APIRouter, Request

from app.constants import EMBED_MODEL, LLM_PROVIDER, COLLECTION_NAME
from app.utils.data_sources import count_flow_docs_on_disk, count_forms_on_disk
router = APIRouter()


@router.get("/health")
def health(request: Request):
    collection = request.app.state.collection
    doc_count = collection.count() if collection is not None else 0
    timestamp = datetime.now().isoformat()
    status = "ok" if collection is not None else "no_index"
    data = {
        "status": status,
        "doc_count": doc_count,
        "form_count": count_forms_on_disk(),
        "flow_doc_count": count_flow_docs_on_disk(skip_empty=True),
        "llm_provider": LLM_PROVIDER,
        "collection_name": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "timestamp": timestamp
    }
    return data
