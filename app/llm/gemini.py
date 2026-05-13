"""Google Gemini provider via the `google-genai` SDK."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.llm.base import ProviderError

logger = logging.getLogger("taxapp.api")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Translate canonical chat messages to Gemini's `contents` shape.

    Gemini uses `role="model"` for assistant turns and wraps text in `parts`.
    System prompts are passed via `GenerateContentConfig.system_instruction`,
    so they are dropped from `contents` here.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        gemini_role = "model" if role == "assistant" else "user"
        out.append({"role": gemini_role, "parts": [{"text": msg.get("content", "")}]})
    return out


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> "GeminiProvider":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured on the server."
            )
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        logger.info("Resolved Gemini model: %s", model)
        return cls(genai.Client(api_key=api_key), model)

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
        contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        response = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def aclose(self) -> None:
        # google-genai's Client manages its own httpx session and exposes no
        # async close hook; rely on process shutdown to release sockets.
        return None
