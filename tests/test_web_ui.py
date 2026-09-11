"""Operator JSON API / SPA-backed flow tests.

The React SPA talks entirely to /api/* JSON endpoints; these tests exercise
them the same way the SPA does.
"""

from __future__ import annotations

import asyncio

from conftest import answer_via_web, find_pending_row, parked_post


async def test_spa_index_served(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<div id=\"root\">" in r.text or "HumanGPT" in r.text


async def test_api_meta(client, state):
    r = await client.get("/api/meta")
    assert r.status_code == 200
    d = r.json()
    assert d["api_base_url"].endswith("/v1")
    assert "pending_count" in d
    assert "human-gpt" in d["models"]


async def test_api_queue_and_detail(client, state):
    body = {
        "model": "human-gpt",
        "messages": [
            {"role": "system", "content": "sys prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsb3=="}},
                ],
            },
        ],
        "tools": [
            {"type": "function", "function": {"name": "frobnicate", "parameters": {"type": "object", "properties": {}}}}
        ],
        "temperature": 0.5,
    }
    await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)

    q = await client.get("/api/requests")
    assert q.status_code == 200
    pending = q.json()["pending"]
    assert len(pending) == 1
    assert pending[0]["id"] == row.id
    assert pending[0]["model"] == "human-gpt"
    assert pending[0]["has_images"] is True

    d = await client.get(f"/api/requests/{row.id}")
    assert d.status_code == 200
    detail = d.json()
    assert detail["body"]["tools"][0]["function"]["name"] == "frobnicate"
    assert detail["body"]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:")
    assert detail["parsed"]["params"]["temperature"] == 0.5


async def test_api_answer_releases_parked(client, state):
    task = await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "audit me"}],
    })
    row = await find_pending_row(state)
    resp = await answer_via_web(client, row.id, text="recorded reply", operator="bob")
    assert resp.status_code == 200
    assert resp.json()["state"] == "answered"
    await task

    fresh = state.db.get_request(row.id)
    assert fresh.state == "answered"
    assert fresh.answered_by == "bob"


async def test_api_templates_crud(client):
    r = await client.post("/api/templates", json={"name": "positive", "body": "Good job!"})
    assert r.status_code == 200
    r = await client.get("/api/templates")
    assert {"name": "positive", "body": "Good job!"} in r.json()["templates"]

    r = await client.post("/api/templates/positive/delete")
    assert r.status_code == 200
    assert (await client.get("/api/templates")).json()["templates"] == []


async def test_api_history(client, state):
    task = await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "audit me"}],
    })
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="recorded reply", operator="bob")
    await task

    r = await client.get("/api/history")
    rows = r.json()["rows"]
    assert any(x["id"] == row.id for x in rows)

    d = await client.get(f"/api/history/{row.id}")
    detail = d.json()
    assert detail["body"]["messages"][0]["content"] == "audit me"
    assert detail["response"]["choices"][0]["message"]["content"] == "recorded reply"
    assert detail["request"]["answered_by"] == "bob"


async def test_api_stream_word(client, state):
    await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "stream": True,
        "messages": [{"role": "user", "content": "typ"}],
    })
    row = await find_pending_row(state)
    r = await client.post(f"/api/requests/{row.id}/stream-word", json={"delta": "Hello "})
    assert r.status_code == 200
    assert r.json()["ok"] is True


async def test_api_responses_string_input_detail(client, state):
    """Step 4 of the SDK quickstart uses a string `input`; the detail endpoint
    must return it intact (the SPA normalizes it defensively in JS)."""
    await parked_post(client, "/v1/responses", {
        "model": "human-gpt",
        "input": "What color is the ocean?",
        "instructions": "Be terse.",
    })
    row = await find_pending_row(state, endpoint="/v1/responses")
    d = await client.get(f"/api/requests/{row.id}")
    assert d.status_code == 200
    body = d.json()["body"]
    assert body["input"] == "What color is the ocean?"
    assert isinstance(body["input"], str)


async def test_api_return_timeout_discard(client, state):
    task = await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "actions"}],
    })
    row = await find_pending_row(state)

    r = await client.post(f"/api/requests/{row.id}/timeout")
    assert r.status_code == 200
    await task  # client got 504
    assert state.db.get_request(row.id).state == "timed_out"

    task2 = await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "actions2"}],
    })
    row2 = await find_pending_row(state)
    r = await client.post(f"/api/requests/{row2.id}/discard")
    assert r.status_code == 200
    await task2  # client got 503
    assert state.db.get_request(row2.id).state == "discarded"


async def test_api_unknown_request(client):
    assert (await client.get("/api/requests/nope")).status_code == 404
    assert (await client.post("/api/requests/nope/answer", json={"text": "x"})).status_code == 404


async def test_claim_accepts_empty_body(client, state):
    """The SPA claims with an empty POST body — must not 400."""
    await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "claim me"}],
    })
    row = await find_pending_row(state)
    r = await client.post(f"/api/requests/{row.id}/claim")
    assert r.status_code == 200
    assert state.db.get_request(row.id).claimed_by == "anonymous"

    r2 = await client.post(
        f"/api/requests/{row.id}/claim", json={"operator": "alice"}
    )
    assert r2.status_code == 200
    assert state.db.get_request(row.id).claimed_by == "alice"


async def test_request_detail_includes_templates(client, state):
    await client.post("/api/templates", json={"name": "positive", "body": "Good job!"})
    await parked_post(client, "/v1/chat/completions", {
        "model": "human-gpt",
        "messages": [{"role": "user", "content": "templates"}],
    })
    row = await find_pending_row(state)
    d = await client.get(f"/api/requests/{row.id}")
    assert d.status_code == 200
    templates = d.json()["templates"]
    assert templates == [{"name": "positive", "body": "Good job!"}]


async def test_live_events_stream(client, state):

    # /events is an SSE stream; with ASGITransport it buffers forever.
    # Exercise the hub path via a real subscription instead.
    from humangpt.sse import format_sse_event

    listener_id, queue = state.hub.subscribe()
    state.hub.publish({"type": "request-changed", "id": "req_x", "state": "pending"})
    event = await asyncio.wait_for(queue.get(), timeout=1)
    state.hub.unsubscribe(listener_id)
    assert event["type"] == "request-changed"
    assert "req_x" in format_sse_event("update", event)
