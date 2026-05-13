"""Retrieval-augmented generation core: embeddings, retrieval, and pipeline.

The LLM-streaming layer lives in `app.llm` and is provider-agnostic — this
package only knows how to build prompts and call `LLMProvider.stream(...)`.
"""
