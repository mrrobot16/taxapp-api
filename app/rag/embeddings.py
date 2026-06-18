"""Embedding model wrapper used at query time by the Chroma collection.

Must match the wrapper used by the indexer exactly so the same vector space
is used for indexing and retrieval.
"""

from functools import lru_cache
from typing import Any, cast

import numpy as np
import torch
from chromadb.api.types import Documents, Embeddable, EmbeddingFunction
from sentence_transformers import SentenceTransformer

from app.constants import EMBED_MODEL


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """Same wrapper used by the indexer — must match exactly."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name, device=_get_device())
        # Cache encodes for individual query strings so repeated questions
        # skip the GPU/MPS encode entirely. Bound the size so we don't grow
        # unboundedly across a long-lived process.
        self._encode_one_cached = lru_cache(maxsize=256)(self._encode_one)

    def _encode_one(self, text: str) -> tuple[float, ...]:
        # Returned as a tuple so the lru_cache value is immutable/hashable-safe;
        # callers convert back to list at the boundary.
        return tuple(self.model.encode([text], show_progress_bar=False)[0].tolist())

    def __call__(self, input: Embeddable) -> list[list[float]]:
        if input and not isinstance(input[0], str):
            raise TypeError("LocalEmbeddingFunction only supports text documents")
        documents = cast(Documents, input)
        if len(documents) == 1:
            return [list(self._encode_one_cached(documents[0]))]
        encoded = np.asarray(self.model.encode(documents, show_progress_bar=False))
        return cast(list[list[float]], encoded.tolist())

    def embed_query(self, input: Embeddable) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: Embeddable) -> list[list[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return EMBED_MODEL

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LocalEmbeddingFunction":
        return LocalEmbeddingFunction(config.get("model_name", EMBED_MODEL))

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self._model_name}
