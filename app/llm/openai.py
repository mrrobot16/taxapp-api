"""OpenAI provider using the async chat-completions streaming API."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.llm.base import ProviderError

logger = logging.getLogger("taxapp.api")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not configured on the server.")
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
        logger.info("Resolved OpenAI model: %s", model)
        return cls(AsyncOpenAI(api_key=api_key), model)

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        # OpenAI accepts the same {"role", "content"} message shape as Anthropic,
        # but takes the system prompt as a leading message rather than a kwarg.
        full_messages = [{"role": "system", "content": system}, *messages]
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result

    async def aclose(self) -> None:
        await self._client.close()
