"""Shared fixtures for the HumanGPT test suite.

The app is served in-process through httpx's ASGITransport, so parked requests
and the web UI share one event loop — exactly how the single-worker server
behaves in production.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from humangpt.app import create_app
from humangpt.config import Settings


@pytest.fixture
def models_json(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "human-gpt", "created": 1715367049, "owned_by": "humangpt"}
                ]
            }
        )
    )
    return path


@pytest.fixture
def settings(tmp_path: Path, models_json: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        models_config=models_json,
        request_timeout_s=60,
        stream_chunk_delay_ms=0,
        stream_mode="word-chunk",
    )


@pytest_asyncio.fixture
async def app(settings: Settings):
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as client:
        yield client


@pytest.fixture
def state(app):
    return app.state.app_state


async def parked_post(client: httpx.AsyncClient, path: str, body: dict):
    """Start a parked API call without blocking the test body."""
    return asyncio.create_task(client.post(path, json=body))


async def answer_via_web(
    client: httpx.AsyncClient,
    request_id: str,
    *,
    text: str = "Hello from the human operator!",
    is_tool_call: bool = False,
    tool_name: str = "",
    tool_arguments: str = "",
    stream_mode: str = "word-chunk",
    operator: str = "tester",
):
    """Submit an answer through the JSON operator API (the SPA's endpoint)."""
    payload = {
        "operator": operator,
        "text": text,
        "is_tool_call": is_tool_call,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "stream_mode": stream_mode,
    }
    return await client.post(f"/api/requests/{request_id}/answer", json=payload)


async def wait_for_pending(state, request_id: str, timeout: float = 2.0) -> None:
    """Poll until the request row shows up as pending (writes are committed fast)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        row = state.db.get_request(request_id)
        if row is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"request {request_id} never appeared as pending; got {row}")


async def find_pending_row(state, *, endpoint: str = "/v1/chat/completions"):
    """Return the single pending row for an endpoint (tests create one at a time).

    Yields to the event loop so the parked task has a chance to create its row.
    """
    for _ in range(200):
        pending = state.db.list_requests(states=["pending"])
        matches = [r for r in pending if r.endpoint == endpoint]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AssertionError(f"expected exactly one pending {endpoint} row, got {len(matches)}")
        await asyncio.sleep(0.01)
    raise AssertionError(f"no pending {endpoint} row appeared in time")
