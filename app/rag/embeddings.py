"""Embedding model wrapper used at query time by the Chroma collection.

Must match the wrapper used by the indexer exactly so the same vector space
is used for indexing and retrieval.
"""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbeddingFunction:
    """Same wrapper used by the indexer — must match exactly."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=_get_device())
        # Cache encodes for individual query strings so repeated questions
        # skip the GPU/MPS encode entirely. Bound the size so we don't grow
        # unboundedly across a long-lived process.
        self._encode_one_cached = lru_cache(maxsize=256)(self._encode_one)

    def _encode_one(self, text: str) -> tuple[float, ...]:
        # Returned as a tuple so the lru_cache value is immutable/hashable-safe;
        # callers convert back to list at the boundary.
        return tuple(self.model.encode([text], show_progress_bar=False)[0].tolist())

    def __call__(self, input: list[str]) -> list[list[float]]:
        if len(input) == 1:
            return [list(self._encode_one_cached(input[0]))]
        return self.model.encode(input, show_progress_bar=False).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def name(self) -> str:
        return EMBED_MODEL
