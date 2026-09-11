"""FastAPI dependency providers shared by API endpoints and the web UI."""

from __future__ import annotations

from fastapi import Request

from .container import AppState


def get_state(request: Request) -> AppState:
    """Resolve the app container from ``app.state.app_state``."""
    return request.app.state.app_state  # type: ignore[attr-defined]
