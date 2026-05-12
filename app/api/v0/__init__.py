"""v0 of the Taxapp API.

Mounted under the `/api` URL prefix so clients continue to hit
`/api/chat` and `/api/health` unchanged. The `v0` package name is for
internal versioning and is intentionally not reflected in the URL —
future versions can be added alongside without breaking existing
clients.
"""

from fastapi import APIRouter

from app.api.v0.chat import router as chat_router
from app.api.v0.health import router as health_router

router = APIRouter(prefix="/api")
router.include_router(chat_router)
router.include_router(health_router)
