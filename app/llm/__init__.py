"""Provider-agnostic LLM streaming layer.

The RAG pipeline depends only on the `LLMProvider` Protocol defined here; the
concrete provider is selected at startup by `build_provider_from_env()` based
on the `LLM_PROVIDER` environment variable.
"""

from app.llm.base import LLMProvider, ProviderError
from app.llm.factory import build_provider_from_env

__all__ = ["LLMProvider", "ProviderError", "build_provider_from_env"]
