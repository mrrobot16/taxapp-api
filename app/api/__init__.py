"""HTTP transport layer.

Aggregates versioned API routers. New versions get mounted here.
"""

from fastapi import APIRouter

from app.api.v0 import router as v0_router

router = APIRouter()
router.include_router(v0_router)
