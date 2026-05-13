"""Anthropic Claude provider.

Owns Claude-specific concerns: API client construction, model resolution
from `ANTHROPIC_MODEL`/`DEFAULT_ANTHROPIC_MODELS`, fallback across the
candidate list when a model is unavailable, and pinning the first model
that succeeds so subsequent requests try it first.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from app.constants import DEFAULT_ANTHROPIC_MODELS
from app.llm.base import ProviderError
from app.utils.logger import Colors

logger = logging.getLogger("taxapp.api")


def resolve_anthropic_models() -> list[str]:
    """Return the ordered list of Claude models to try, env-override first."""
    configured = os.getenv("ANTHROPIC_MODEL", "").strip()
    models = [configured] + DEFAULT_ANTHROPIC_MODELS if configured else DEFAULT_ANTHROPIC_MODELS[:]
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    logger.info("Resolved Anthropic models: %s", deduped)
    return deduped


def is_anthropic_access_error(err: BaseException) -> bool:
    """True for SDK errors that mean 'try a different model'."""
    return isinstance(err, (anthropic.NotFoundError, anthropic.PermissionDeniedError))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic, models: list[str]) -> None:
        self._client = client
        self._models = models

    @classmethod
    def from_env(cls) -> "AnthropicProvider":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured on the server.")
        return cls(AsyncAnthropic(api_key=api_key), resolve_anthropic_models())

    @property
    def models(self) -> list[str]:
        """Read-only view of the current candidate list (winner-first after pin)."""
        return list(self._models)

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        if not self._models:
            raise ProviderError(
                "No Anthropic models configured. Set ANTHROPIC_MODEL in .env "
                "to a model your API key can access."
            )

        last_error: BaseException | None = None
        streamed = False
        chosen_model: str | None = None

        for model in list(self._models):
            try:
                logger.info("Trying model: %s%s%s", Colors.MAGENTA, model, Colors.RESET)
                async with self._client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    streamed = True
                    chosen_model = model
                    async for chunk in stream.text_stream:
                        yield chunk
                break
            except Exception as e:
                last_error = e
                # Only roll over to the next candidate if we never committed to
                # this model. Once we've started streaming, errors propagate.
                if not streamed and is_anthropic_access_error(e):
                    continue
                raise

        if not streamed:
            msg = (
                "No allowed Anthropic model found for this API key. "
                "Set ANTHROPIC_MODEL in .env to a model your key can access."
            )
            if last_error is not None:
                msg = f"{msg} ({last_error})"
            raise ProviderError(msg)

        # Pin the winning model so subsequent calls skip dead candidates.
        if chosen_model is not None and self._models[0] != chosen_model:
            self._models = [chosen_model] + [m for m in self._models if m != chosen_model]

    async def aclose(self) -> None:
        await self._client.close()
