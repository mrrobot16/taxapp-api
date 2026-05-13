"""Pick and construct the active LLM provider.

The active provider is selected by `app.constants.LLM_PROVIDER`; per-provider
env vars (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) supply
credentials. Providers are imported lazily so a missing optional SDK only
fails the run when that provider is actually selected.
"""

from __future__ import annotations

import logging

from app.constants import LLM_PROVIDER, SUPPORTED_PROVIDERS
from app.llm.base import LLMProvider, ProviderError

logger = logging.getLogger("taxapp.api")

def build_provider_from_env() -> LLMProvider:
    """Construct the configured `LLMProvider`, or raise `ProviderError`."""
    logger.info("Selected LLM provider: %s", LLM_PROVIDER)

    if LLM_PROVIDER == "anthropic":
        from app.llm.anthropic import AnthropicProvider

        return AnthropicProvider.from_env()
    if LLM_PROVIDER == "openai":
        from app.llm.openai import OpenAIProvider

        return OpenAIProvider.from_env()
    if LLM_PROVIDER == "gemini":
        from app.llm.gemini import GeminiProvider

        return GeminiProvider.from_env()

    raise ProviderError(
        f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )
