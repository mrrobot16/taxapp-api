"""Anthropic model resolution and error classification helpers."""

import logging
import os

import anthropic

from app.constants import DEFAULT_ANTHROPIC_MODELS

logger = logging.getLogger("taxapp.api")


def get_candidate_models() -> list[str]:
    configured = os.getenv("ANTHROPIC_MODEL", "").strip()
    models = [configured] + DEFAULT_ANTHROPIC_MODELS if configured else DEFAULT_ANTHROPIC_MODELS[:]
    logger.debug("Candidate models: %s", models)
    seen = set()
    deduped = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    logger.info("Resolved Anthropic models: %s", deduped)
    return deduped


def is_model_access_error(err: Exception) -> bool:
    return isinstance(err, (anthropic.NotFoundError, anthropic.PermissionDeniedError))
