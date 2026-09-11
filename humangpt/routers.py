"""Router registry + resetable FastAPI router.

The OpenAI-compatible surface is mounted under a single ``/v1`` APIRouter
in ``humangpt/app.py``. Adding a new endpoint later (e.g. ``/v1/embeddings``)
is a small, isolated change: write a module under ``humangpt/endpoints/``
with ``router = APIRouter()``, register a handler, and add one include line
here. See README "how to add a new /v1 endpoint later".
"""

from __future__ import annotations

from fastapi import APIRouter

from .endpoints import chat, models, responses

v1_router = APIRouter()
v1_router.include_router(models.router)
v1_router.include_router(chat.router)
v1_router.include_router(responses.router)
