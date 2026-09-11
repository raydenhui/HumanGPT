"""Vision and parsing edge cases for chat completions."""

from __future__ import annotations

import json

from conftest import answer_via_web, find_pending_row, parked_post


async def test_chat_vision_data_uri(client, state):
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSAA=="
    body = {
        "model": "human-gpt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                ],
            }
        ],
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    assert row.parsed_json["has_images"] is True
    await answer_via_web(client, row.id, text="It contains a tiny PNG.")
    resp = await task
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "It contains a tiny PNG."


async def test_chat_vision_http_url(client, state):
    body = {
        "model": "human-gpt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What brand is this logo?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/logo.png"}},
                ],
            }
        ],
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    assert row.parsed_json["has_images"] is True
    await answer_via_web(client, row.id, text="Some logo, probably.")
    resp = await task
    assert resp.status_code == 200


async def test_chat_vision_string_image_url_form(client, state):
    # SDKs also send image_url as a bare string in the part.
    body = {
        "model": "human-gpt",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": "https://example.com/pic.png"}],
            }
        ],
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="seen it")
    resp = await task
    assert resp.status_code == 200


async def test_developer_and_tool_roles_accepted(client, state):
    body = {
        "model": "human-gpt",
        "messages": [
            {"role": "developer", "content": "You are precise."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": json.dumps({"result": 42}),
            },
        ],
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="noted")
    resp = await task
    assert resp.status_code == 200
