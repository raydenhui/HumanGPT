"""GET /v1/models — the configured model list.

Extension pattern: every /v1 group lives in its own module with a module-level
``router = APIRouter()`` and is included in ``humangpt/routers.py``. Adding
``/v1/embeddings`` later means a new module plus one include line.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse

from ..container import AppState
from ..deps import get_state

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(
    request: Request,
    state: AppState = Depends(get_state),
) -> JSONResponse:
    """Return the merged model catalog in the OpenAI envelope.

    Each model includes optional user-editable ``description`` and ``pricing``
    (from the Settings panel) alongside the file-defined identity fields.
    """
    return JSONResponse({"object": "list", "data": state.models})
