"""Responses API: non-streaming envelope, streaming event order, vision input."""

from __future__ import annotations

import json

from conftest import answer_via_web, find_pending_row, parked_post

RESP_BODY = {
    "model": "human-gpt",
    "instructions": "You are the QA oracle. Answer tersely.",
    "input": [
        {"type": "message", "role": "user", "content": "Is the sky blue?"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me think about it carefully."}],
        },
    ],
    "max_output_tokens": 256,
    "temperature": 0.0,
    "seed": 7,
}


async def test_responses_non_streaming(client, state):
    task = await parked_post(client, "/v1/responses", RESP_BODY)
    row = await find_pending_row(state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="Yes. The sky appears blue on a clear day.")

    resp = await task
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["model"] == "human-gpt"
    assert body["status"] == "completed"
    assert body["id"].startswith("resp-")
    assert isinstance(body["created_at"], int)
    assert len(body["output"]) == 1
    out = body["output"][0]
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"][0]["type"] == "output_text"
    assert out["content"][0]["text"] == "Yes. The sky appears blue on a clear day."
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0


async def test_responses_string_input(client, state):
    task = await parked_post(
        client, "/v1/responses", {"model": RESP_BODY["model"], "input": "just a string"}
    )
    row = await find_pending_row(state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="string received")
    resp = await task
    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "string received"


async def test_responses_streaming_event_order(client, state):
    """Responses API SSE event order in once mode (single content delta)."""
    task = await parked_post(client, "/v1/responses", {**RESP_BODY, "stream": True, "stream_mode": "once"})
    row = await find_pending_row(state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="Green, or sometimes orange.", stream_mode="once")

    resp = await task
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = resp.text.strip().split("\n\n")
    events = []
    for frame in lines:
        event_name = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append((event_name, data))

    names = [e_name for e_name, _ in events]
    # exact order contract: prefix, exactly 1 delta (once), suffix, completed.
    expected = [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert names == expected

    deltas = [d["delta"] for name, d in events if name == "response.output_text.delta"]
    assert deltas == ["Green, or sometimes orange."]

    completed = [d for name, d in events if name == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["output"][0]["content"][0]["text"] == "Green, or sometimes orange."


async def test_responses_live_word_chunk_stream(client, state):
    """Word-chunk mode: words pushed live via /stream-word appear as deltas
    in the canonical order."""
    task = await parked_post(client, "/v1/responses", {**RESP_BODY, "stream": True})
    row = await find_pending_row(state, endpoint="/v1/responses")
    for w in ["How ", "now ", "brown ", "cow."]:
        r = await client.post(f"/api/requests/{row.id}/stream-word", json={"delta": w})
        assert r.status_code == 200
    await answer_via_web(client, row.id, text="How now brown cow.", stream_mode="word-chunk")

    resp = await task
    frames = resp.text.strip().split("\n\n")
    names = []
    deltas = []
    for frame in frames:
        name = next(
            (line[len("event: "):] for line in frame.split("\n") if line.startswith("event: ")),
            None,
        )
        data = next(
            (json.loads(line[len("data: "):]) for line in frame.split("\n") if line.startswith("data: ")),
            None,
        )
        names.append(name)
        if name == "response.output_text.delta":
            deltas.append(data["delta"])
    assert names[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert "".join(deltas) == "How now brown cow."
    assert names[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]


async def test_responses_vision_input_image(client, state):
    body = {
        "model": "human-gpt",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
                        "detail": "high",
                    },
                ],
            }
        ],
    }
    task = await parked_post(client, "/v1/responses", body)
    row = await find_pending_row(state, endpoint="/v1/responses")
    assert row.parsed_json["has_images"] is True
    await answer_via_web(client, row.id, text="A red circle on a white background.")
    resp = await task
    assert resp.status_code == 200


async def test_responses_streaming_whole_message(client, state):
    task = await parked_post(client, "/v1/responses", {**RESP_BODY, "stream": True})
    row = await find_pending_row(state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="one burst", stream_mode="once")
    resp = await task
    frames = resp.text.strip().split("\n\n")
    delta_frames = [f for f in frames if '"response.output_text.delta"' in f]
    assert len(delta_frames) == 1
    assert '"one burst"' in delta_frames[0]
