"""Pick and construct the active LLM provider from environment variables.

`LLM_PROVIDER` selects the implementation; per-provider env vars (e.g.
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) supply credentials.
Providers are imported lazily so a missing optional SDK only fails the run
when that provider is actually selected.
"""

from __future__ import annotations

import logging
import os

from app.llm.base import LLMProvider, ProviderError
from app.constants import LLM_PROVIDER
logger = logging.getLogger("taxapp.api")

DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("anthropic", "openai", "gemini")


def build_provider_from_env() -> LLMProvider:
    """Construct the configured `LLMProvider`, or raise `ProviderError`."""
    # NOTE: We are using a hardcoded LLM provider in the constants file, instead of using the environment variable.
    # name = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
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
