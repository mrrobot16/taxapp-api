"""Cached access to the Chroma vector store."""

import logging
import threading

import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
from app.rag.embeddings import LocalEmbeddingFunction

logger = logging.getLogger("taxapp.api")

_collection = None
_lock = threading.Lock()


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    with _lock:
        if _collection is not None:
            return _collection
        if not CHROMA_DIR.exists():
            return None
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            embed_fn = LocalEmbeddingFunction(EMBED_MODEL)
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
            )
            return _collection
        except Exception as e:
            logger.error("Could not load collection: %s", e)
            return None


def warmup_collection():
    """Eagerly initialize the Chroma collection + embedding model.

    Intended for the FastAPI startup lifespan so the first chat request
    doesn't pay the SentenceTransformer load + first-encode cost.
    """
    return get_collection()
