"""LLM provider interface shared by all concrete vendor implementations.

The RAG pipeline depends only on this Protocol; swapping providers is a
configuration change, not a code change. Each implementation is responsible
for translating the canonical message shape into its vendor SDK's format and
for handling vendor-specific concerns like model fallback.
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Terminal error from an LLM provider that the pipeline should surface.

    Used when a provider exhausts its configured models, lacks credentials,
    or encounters any other failure that cannot be recovered from internally.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Streaming text generator for a single conversation turn.

    Implementations must:
      - Accept the canonical message shape `[{"role": "user"|"assistant", "content": str}]`.
      - Yield plain text chunks as they arrive from the upstream SDK.
      - Raise `ProviderError` (or let a SDK exception propagate) on failure.
    """

    name: str

    def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        ...

    async def aclose(self) -> None:
        ...
