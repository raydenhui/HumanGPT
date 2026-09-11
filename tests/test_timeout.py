"""Timeout behaviour and restart/interrupt semantics."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from conftest import answer_via_web, find_pending_row

from humangpt.endpoint_utils import build_request_row
from humangpt.summary import summarize_chat

CHAT_BODY = {
    "model": "human-gpt",
    "messages": [{"role": "user", "content": "slow operator?"}],
}


async def test_timeout_releases_with_504(settings, tmp_path, models_json):
    from humangpt.app import create_app

    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        state = app.state.app_state
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30
        ) as client:
            task = asyncio.create_task(
                client.post("/v1/chat/completions", json={**CHAT_BODY, "stream": False})
            )
            await asyncio.sleep(0.1)
            row = state.db.list_requests(states=["pending"])[0]
            # force-sweep by making the deadline already past
            state.db._execute(
                "UPDATE requests SET timeout_at = ? WHERE id = ?",
                (time.time() - 1, row.id),
            )
            state.db.commit()
            state.queue.sweep_expired()

            resp = await task
            assert resp.status_code == 504
            error = resp.json()["error"]
            assert error["type"] == "server_error"
            assert error["code"] == "request_timeout"
            assert "timed out" in error["message"]

            fresh = state.db.get_request(row.id)
            assert fresh.state == "timed_out"


async def test_startup_sweep_marks_orphaned_pending_interrupted(settings):
    from humangpt.app import create_app

    # Simulate a crashed process: a request row exists in 'pending' with no
    # waiter (create it directly, then boot a fresh app over the same DB).
    from humangpt.db import Database

    db = Database(settings.db_path)
    row = build_request_row(
        endpoint="/v1/chat/completions",
        model="m",
        body_raw=CHAT_BODY,
        parsed_summary=summarize_chat(CHAT_BODY),
        stream_mode="word-chunk",
        timeout_s=600,
    )
    db.create_request(row)
    db.close()

    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        fresh = app.state.app_state.db.get_request(row.id)
        assert fresh.state == "interrupted"
        assert fresh.interrupted_at is not None


async def test_manual_discard_releases_client_with_503(client, state):
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT_BODY))
    await asyncio.sleep(0.1)
    row = state.db.list_requests(states=["pending"])[0]
    act = await client.post(f"/api/requests/{row.id}/discard")
    assert act.status_code == 200
    resp = await task
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "request_discarded"


async def test_manual_timeout_action(client, state):
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT_BODY))
    await asyncio.sleep(0.1)
    row = state.db.list_requests(states=["pending"])[0]
    act = await client.post(f"/api/requests/{row.id}/timeout")
    assert act.status_code == 200
    resp = await task
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "request_timeout"


async def test_return_to_pending_after_answer(client, state):
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT_BODY))
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="first take")
    await task  # client got the first answer

    act = await client.post(f"/api/requests/{row.id}/return")
    assert act.status_code == 200
    fresh = state.db.get_request(row.id)
    assert fresh.state == "pending"
    assert fresh.timeout_at is not None  # timeout window re-armed

    # answering the returned request again works and yields a second response row
    await answer_via_web(client, row.id, text="second take")
    fresh = state.db.get_request(row.id)
    assert fresh.state == "answered" or fresh.state == "streamed"
    rows = state.db._execute("SELECT * FROM responses WHERE request_id = ?", (row.id,)).fetchall()
    assert len(rows) == 2
    bodies = [json.loads(r["body"]) for r in rows]
    assert {"first take", "second take"} == {
        b["choices"][0]["message"]["content"] for b in bodies
    }


async def test_double_submit_second_gets_409(client, state):
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT_BODY))
    await asyncio.sleep(0.1)
    row = state.db.list_requests(states=["pending"])[0]

    from conftest import answer_via_web as aw

    first = await aw(client, row.id, text="winner")
    assert first.status_code == 200
    second = await aw(client, row.id, text="loser")
    assert second.status_code == 409
    await task

    rows = state.db._execute("SELECT * FROM responses WHERE request_id = ?", (row.id,)).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["body"])["choices"][0]["message"]["content"] == "winner"


async def test_unknown_request_id(client):
    assert (await client.get("/api/requests/nope")).status_code == 404
    r = await client.post("/api/requests/nope/answer", json={"text": "x"})
    assert r.status_code == 404
