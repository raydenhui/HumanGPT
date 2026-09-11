"""Chat completions: parsing, non-streaming answer flow, tool calls, streams."""

from __future__ import annotations

import json

from conftest import answer_via_web, find_pending_row, parked_post, wait_for_pending

CHAT_BODY = {
    "model": "human-gpt",
    "messages": [
        {"role": "system", "content": "You are a helpful QA assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    "max_tokens": 64,
    "temperature": 0.2,
    "top_p": 0.9,
    "seed": 42,
    "stop": ["\n@done", "\n"],
    "presence_penalty": 0.1,
    "frequency_penalty": 0.1,
    "logprobs": True,
    "top_logprobs": 5,
    "response_format": {"type": "text"},
    "metadata": {"session": "qa-session-1"},
    "user": "qa-runner",
}


async def test_non_streaming_roundtrip(client, state):
    task = await parked_post(client, "/v1/chat/completions", CHAT_BODY)
    row = await find_pending_row(state)
    await wait_for_pending(state, row.id)

    act = await answer_via_web(client, row.id, text="The answer is four.")
    assert act.status_code == 200

    resp = await task
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "human-gpt"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["system_fingerprint"]
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "The answer is four."
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # The state machine recorded the answer.
    fresh = state.db.get_request(row.id)
    assert fresh.state == "answered"
    assert fresh.answered_by == "tester"
    fresh_response = state.db.get_response_for_request(row.id)
    assert fresh_response.body_json["choices"][0]["message"]["content"] == "The answer is four."


async def test_streaming_sequence_and_reconstruction(client, state):
    """Word-chunk live streaming: words pushed via /stream-word, then finished.

    The terminal stream (finish_reason + [DONE]) arrives after the operator
    answers/submits; the earlier words already streamed via /stream-word must
    appear as content deltas in the SSE body the parked client receives.
    """
    task = await parked_post(
        client,
        "/v1/chat/completions",
        {**CHAT_BODY, "stream": True, "stream_options": {"include_usage": True}},
    )
    row = await find_pending_row(state)

    # the operator typo "Hello world, this is streamed." locks word by word
    words = ["Hello ", "world, ", "this ", "is ", "streamed."]
    for w in words:
        r = await client.post(f"/api/requests/{row.id}/stream-word", json={"delta": w})
        assert r.status_code == 200
    # then submit
    await answer_via_web(client, row.id, text="".join(words))

    resp = await task
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = resp.text.strip().split("\n\n")
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]

    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == "human-gpt" for c in chunks)

    reconstructed = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks
    )
    assert reconstructed == "Hello world, this is streamed."

    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] > 0


async def test_stream_whole_message_mode(client, state):
    """once mode: submit, then the whole message is a single content delta."""
    task = await parked_post(
        client, "/v1/chat/completions", {**CHAT_BODY, "stream": True, "stream_mode": "once"}
    )
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="single burst", stream_mode="once")

    resp = await task
    frames = resp.text.strip().split("\n\n")
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    content_deltas = [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c["choices"][0]["delta"].get("content") is not None
    ]
    assert content_deltas == ["single burst"]


async def test_tool_call_answer(client, state):
    body = {
        **CHAT_BODY,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    await answer_via_web(
        client,
        row.id,
        text="",
        is_tool_call=True,
        tool_name="get_weather",
        tool_arguments='{"city": "Paris"}',
    )

    resp = await task
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


async def test_streaming_partial_push_reconciles_remainder(client, state):
    """If the caller streams only part of the answer, the endpoint emits the
    remainder so the client's accumulated text equals the submitted envelope."""
    task = await parked_post(
        client, "/v1/chat/completions", {**CHAT_BODY, "stream": True}
    )
    row = await find_pending_row(state)
    # stream "1", " 2", " 3" live, but submit the full "1 2 3 4 5"
    for w in ["1", " 2", " 3"]:
        assert (await client.post(f"/api/requests/{row.id}/stream-word", json={"delta": w})).status_code == 200
    await answer_via_web(client, row.id, text="1 2 3 4 5", stream_mode="word-chunk")

    resp = await task
    frames = resp.text.strip().split("\n\n")
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    reconstructed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert reconstructed == "1 2 3 4 5"


async def test_non_streaming_state_is_answered(client, state):
    """Non-streaming answers must land in `answered` (not `streamed`)."""
    task = await parked_post(client, "/v1/chat/completions", {**CHAT_BODY, "stream": False})
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="plain", stream_mode="word-chunk")
    await task
    assert state.db.get_request(row.id).state == "answered"


async def test_streaming_state_is_streamed(client, state):
    """Streaming answers must land in `streamed`."""
    task = await parked_post(client, "/v1/chat/completions", {**CHAT_BODY, "stream": True})
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="live", stream_mode="word-chunk")
    await task
    assert state.db.get_request(row.id).state == "streamed"


async def test_non_streaming_repeat_submit_rejected(client, state):
    """After a non-streaming answer, a repeat submit is a 409 (not pending)."""
    task = await parked_post(client, "/v1/chat/completions", {**CHAT_BODY, "stream": False})
    row = await find_pending_row(state)
    first = await answer_via_web(client, row.id, text="first")
    assert first.status_code == 200
    second = await answer_via_web(client, row.id, text="second")
    assert second.status_code == 409
    await task


async def test_streaming_tool_call(client, state):
    body = {
        **CHAT_BODY,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    }
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    await answer_via_web(
        client, row.id, text="", is_tool_call=True, tool_name="search", tool_arguments='{"q": "humans"}'
    )

    resp = await task
    frames = resp.text.strip().split("\n\n")
    chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "search"
    arguments = "".join(
        c["choices"][0]["delta"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", "")
        for c in chunks
    )
    assert json.loads(arguments) == {"q": "humans"}


async def test_unknown_fields_are_tolerated_like_openai(client, state):
    body = {**CHAT_BODY, "n": 1, "store": True, "totally_unknown_field": "ignored"}
    task = await parked_post(client, "/v1/chat/completions", body)
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="ok")
    resp = await task
    assert resp.status_code == 200


async def test_malformed_body_rejected(client):
    cases = [
        ({"messages": []}, "missing model"),
        ({"model": "x"}, "missing messages"),
        ({"model": "", "messages": []}, "empty model"),
        ({"model": "x", "messages": [{"role": "wizard", "content": "hi"}]}, "bad role"),
        ({"model": "x", "messages": [{"role": "user"}]}, "user without content"),
        ("not json at all", "not json"),
    ]
    for body, label in cases:
        payload = body if isinstance(body, str) else json.dumps(body)
        r = await client.post(
            "/v1/chat/completions", content=payload, headers={"Content-Type": "application/json"}
        )
        assert r.status_code in (400, 422), label
        error = r.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "message" in error


async def test_usage_heuristic_reasonable(client, state):
    long_body = {
        **CHAT_BODY,
        "messages": [{"role": "user", "content": "word " * 200}],
    }
    task = await parked_post(client, "/v1/chat/completions", long_body)
    row = await find_pending_row(state)
    await answer_via_web(client, row.id, text="short reply")
    resp = await task
    usage = resp.json()["usage"]
    # ~4 bytes per token for the 200-word payload (1200 chars total)
    assert 200 <= usage["prompt_tokens"] <= 1200
    assert usage["completion_tokens"] >= 1
