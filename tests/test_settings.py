"""Settings panel: model catalog merge (models.json + models_meta DB) and
stream-defaults persistence. Exercises the /api/settings + /api/models JSON
endpoints and the merged /v1/models output.
"""

from __future__ import annotations

import json

from humangpt.catalog import load_ui_settings


async def test_models_meta_description_pricing_served(client, state):
    # base file has human-gpt; add description+pricing via the settings API
    r = await client.post(
        "/api/models",
        json={
            "model_id": "human-gpt",
            "description": "Human-operated QA model",
            "pricing": '{"input": 1, "output": 2, "unit": "USD/1K tokens"}',
        },
    )
    assert r.status_code == 200

    # catalog reloaded in-process -> /v1/models shows the edits
    r = await client.get("/v1/models")
    payload = r.json()
    by_id = {m["id"]: m for m in payload["data"]}
    assert by_id["human-gpt"]["description"] == "Human-operated QA model"
    assert by_id["human-gpt"]["pricing"] == {"input": 1, "output": 2, "unit": "USD/1K tokens"}


async def test_new_model_id_added_via_settings(client, state):
    r = await client.post(
        "/api/models",
        json={"model_id": "claude-human", "description": "added via panel"},
    )
    assert r.status_code == 200
    r = await client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert "claude-human" in ids


async def test_settings_endpoint(client, state):
    r = await client.get("/api/settings")
    assert r.status_code == 200
    d = r.json()
    assert d["api_base_url"].endswith("/v1")
    assert d["configured_key"] is False
    assert any(m["id"] == "human-gpt" for m in d["models"])


async def test_stream_settings_persist(client, state):
    r = await client.post(
        "/api/settings", json={"stream_mode": "once", "stream_chunk_delay_ms": 123}
    )
    assert r.status_code == 200
    ui = load_ui_settings(state.db, state.settings)
    assert ui == {"stream_mode": "once", "stream_chunk_delay_ms": 123}
    assert state.db.get_setting("stream_mode") == "once"


async def test_stream_setting_governs_new_chat_requests(client, state):
    # set once-mode globally, then a stream chat without stream_mode must
    # resolve with a single content delta
    await client.post("/api/settings", json={"stream_mode": "once", "stream_chunk_delay_ms": 0})

    import asyncio

    task = asyncio.create_task(
        client.post(
            "/v1/chat/completions",
            json={
                "model": "human-gpt",
                "stream": True,
                "messages": [{"role": "user", "content": "global setting applies"}],
            },
        )
    )
    from conftest import find_pending_row

    row = await find_pending_row(state)
    from conftest import answer_via_web

    await answer_via_web(client, row.id, text="global once", stream_mode="")

    resp = await task
    frames = resp.text.strip().split("\n\n")
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    deltas = [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c["choices"][0]["delta"].get("content") is not None
    ]
    assert deltas == ["global once"]
