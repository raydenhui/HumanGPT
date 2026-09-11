"""Web entry: serves the built React SPA (humangpt/web/static/dist).

The SPA talks to /api/* (humangpt/api/routes.py) and /events (SSE live
updates). The old server-rendered Jinja templates are superseded by the SPA;
these routes only provide the SPA shell + the live-event stream.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

from ..container import AppState
from ..deps import get_state
from ..sse import format_sse_event

router = APIRouter()

_DIST = Path(__file__).parent / "static" / "dist"
_INDEX = _DIST / "index.html"


def _serve_spa() -> HTMLResponse:
    return HTMLResponse(_INDEX.read_text(encoding="utf-8"))


@router.get("/", response_class=HTMLResponse)
async def spa_index():
    return _serve_spa()


@router.get("/requests/{request_id}", response_class=HTMLResponse)
async def spa_request():
    return _serve_spa()


@router.get("/history", response_class=HTMLResponse)
async def spa_history():
    return _serve_spa()


@router.get("/history/{request_id}", response_class=HTMLResponse)
async def spa_history_detail():
    return _serve_spa()


@router.get("/templates", response_class=HTMLResponse)
async def spa_templates():
    return _serve_spa()


@router.get("/settings", response_class=HTMLResponse)
async def spa_settings():
    return _serve_spa()


@router.get("/events")
async def events(request: Request, state: AppState = Depends(get_state)):
    """SSE: live queue updates for the SPA (one connection per browser tab)."""

    async def _stream():
        listener_id, queue = state.hub.subscribe()
        keepalive_s = max(1.0, float(state.settings.ui_sse_keepalive_s))
        try:
            yield format_sse_event("connected", {"type": "connected", "ts": int(time.time())})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_s)
                    yield format_sse_event("update", event)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            state.hub.unsubscribe(listener_id)

    return StreamingResponse(_stream(), media_type="text/event-stream")
