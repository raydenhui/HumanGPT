"""Integration with the official openai-python SDK.

These tests point the real SDK's ``base_url`` at the in-process server
(via an injected httpx transport) and exercise the three documented flows:
normal chat, streaming chat (deltas -> [DONE]), and a tool call whose answer
was typed by a human in the UI. Plus the Responses API (non-stream + stream).

They are the proof of SDK compatibility required by the spec.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest_asyncio
from conftest import answer_via_web, find_pending_row
from openai import AsyncOpenAI


@pytest_asyncio.fixture
async def sdk(client: httpx.AsyncClient):
    """An AsyncOpenAI configured exactly like a QA team would point it:
    api_key is arbitrary (any-mode), base_url is this server's /v1.
    """
    return AsyncOpenAI(
        api_key="sk-test-does-not-matter",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=client._transport, base_url="http://test", timeout=30),
    )


async def test_sdk_normal_chat(sdk, client):
    task = asyncio.create_task(
        sdk.chat.completions.create(
            model="human-gpt",
            messages=[
                {"role": "system", "content": "You answer in poetry."},
                {"role": "user", "content": "Say hello."},
            ],
        )
    )
    row = await find_pending_row(client._transport.app.state.app_state)
    await answer_via_web(client, row.id, text="Hello, brave tester!")

    completion = await task
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Hello, brave tester!"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.total_tokens > 0


async def test_sdk_streaming_chat(sdk, client):
    task = asyncio.create_task(
        sdk.chat.completions.create(
            model="human-gpt",
            messages=[{"role": "user", "content": "Count to three."}],
            stream=True,
            extra_body={"stream_mode": "once"},
        )
    )
    row = await find_pending_row(client._transport.app.state.app_state)
    await answer_via_web(client, row.id, text="One, two, three, done!", stream_mode="once")

    stream = await task
    pieces: list[str] = []
    finish_reasons = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            pieces.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)
    assert "".join(pieces) == "One, two, three, done!"
    assert finish_reasons and finish_reasons[-1] == "stop"


async def test_sdk_tool_call_answer(sdk, client):
    task = asyncio.create_task(
        sdk.chat.completions.create(
            model="human-gpt",
            messages=[{"role": "user", "content": "Weather in Berlin?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
    )
    row = await find_pending_row(client._transport.app.state.app_state)
    await answer_via_web(
        client,
        row.id,
        text="",
        is_tool_call=True,
        tool_name="get_weather",
        tool_arguments='{"city": "Berlin"}',
    )

    completion = await task
    call = completion.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Berlin"}
    assert completion.choices[0].finish_reason == "tool_calls"


async def test_sdk_responses_api(sdk, client):
    task = asyncio.create_task(
        sdk.responses.create(
            model="human-gpt",
            instructions="Be terse.",
            input="What color is the ocean?",
        )
    )
    row = await find_pending_row(client._transport.app.state.app_state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="Blue, mostly.")

    resp = await task
    assert resp.object == "response"
    text = "".join(
        part.text
        for item in resp.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )
    assert text == "Blue, mostly."


async def test_sdk_responses_stream(sdk, client):
    task = asyncio.create_task(
        sdk.responses.create(
            model="human-gpt",
            input="tell me something",
            stream=True,
            extra_body={"stream_mode": "once"},
        )
    )
    row = await find_pending_row(client._transport.app.state.app_state, endpoint="/v1/responses")
    await answer_via_web(client, row.id, text="The answer is 42, obviously.", stream_mode="once")

    stream = await task
    event_types: list[str] = []
    delta_texts: list[str] = []
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_text.delta":
            delta_texts.append(event.delta)

    assert "response.created" in event_types
    assert "response.in_progress" in event_types
    assert "response.completed" in event_types
    assert "".join(delta_texts) == "The answer is 42, obviously."
    assert event_types.index("response.completed") == len(event_types) - 1
