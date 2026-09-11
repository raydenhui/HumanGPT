"""GET /v1/models — envelope and config-driven list."""

from __future__ import annotations

import httpx


async def test_models_list_shape(client: httpx.AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    payload = r.json()
    assert payload["object"] == "list"
    data = payload["data"]
    assert len(data) == 1
    ids = {m["id"] for m in data}
    assert ids == {"human-gpt"}
    for m in data:
        assert m["object"] == "model"
        assert isinstance(m["created"], int)
        assert m["owned_by"] == "humangpt"


async def test_models_requires_no_auth_in_any_mode(client: httpx.AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
