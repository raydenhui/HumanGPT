"""Unit tests for humangpt/respond.py — wrapping a human answer in an envelope."""

from __future__ import annotations

import json

from humangpt.db import RequestRow
from humangpt.respond import (
    AnswerPayload,
    build_chat_response,
    build_responses_envelope,
    envelope_text,
    is_tool_call_envelope,
    tool_call_parts,
)


def make_row(*, endpoint="/v1/chat/completions", parsed=None) -> RequestRow:
    return RequestRow(
        id="req_abcdef123456",
        endpoint=endpoint,
        model="human-gpt",
        body=json.dumps({"stream": False}),
        parsed=json.dumps(parsed if parsed is not None else {"prompt_tokens": 12}),
        state="pending",
        created_at=1_700_000_000.0,
        answered_at=None,
        answered_by=None,
        claimed_by=None,
        claimed_at=None,
        interrupted_at=None,
        stream_mode="word-chunk",
        timeout_at=None,
    )


class TestAnswerPayload:
    def test_arguments_text_default(self):
        assert AnswerPayload().arguments_text == "{}"

    def test_arguments_text_passthrough(self):
        assert AnswerPayload(tool_arguments='{"a":1}').arguments_text == '{"a":1}'

    def test_arguments_text_dict_serialized(self):
        assert AnswerPayload(tool_arguments={"a": 1}).arguments_text == '{"a":1}'


class TestBuildChatResponse:
    def test_text_answer(self):
        completion, record = build_chat_response(make_row(), AnswerPayload(text="hello"))
        env = completion.model_dump()
        assert env["object"] == "chat.completion"
        assert env["model"] == "human-gpt"
        assert env["choices"][0]["finish_reason"] == "stop"
        assert env["choices"][0]["message"]["content"] == "hello"
        assert env["usage"]["prompt_tokens"] == 12
        assert env["usage"]["total_tokens"] == 12 + env["usage"]["completion_tokens"]
        assert record["is_tool_call"] is False
        assert record["stream_mode"] == "word-chunk"

    def test_tool_call_answer(self):
        completion, record = build_chat_response(
            make_row(),
            AnswerPayload(is_tool_call=True, tool_name="get_weather", tool_arguments='{"city":"Paris"}'),
        )
        env = completion.model_dump()
        choice = env["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        call = choice["message"]["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        assert record["is_tool_call"] is True
        assert record["tool_name"] == "get_weather"

    def test_prompt_tokens_default_when_missing(self):
        row = make_row(parsed={})
        completion, _ = build_chat_response(row, AnswerPayload(text="x"))
        assert completion.model_dump()["usage"]["prompt_tokens"] == 1

    def test_stream_mode_sanitized(self):
        _, record = build_chat_response(make_row(), AnswerPayload(text="x", stream_mode="bogus"))
        assert record["stream_mode"] == "word-chunk"


class TestBuildResponsesEnvelope:
    def test_text_answer(self):
        env_obj, record = build_responses_envelope(make_row(endpoint="/v1/responses"), AnswerPayload(text="hi"))
        env = env_obj.model_dump()
        assert env["object"] == "response"
        out = env["output"][0]
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["content"][0]["text"] == "hi"
        assert env["usage"]["input_tokens"] == 12
        assert record["is_tool_call"] is False

    def test_tool_answer(self):
        env_obj, _ = build_responses_envelope(
            make_row(endpoint="/v1/responses"),
            AnswerPayload(is_tool_call=True, tool_name="search", tool_arguments='{"q":"x"}'),
        )
        out = env_obj.model_dump()["output"][0]
        assert out["type"] == "function_call"
        assert out["name"] == "search"
        assert out["arguments"] == '{"q":"x"}'


class TestEnvelopeIntrospection:
    def test_envelope_text_chat(self):
        completion, _ = build_chat_response(make_row(), AnswerPayload(text="hello"))
        assert envelope_text(completion.model_dump()) == "hello"

    def test_envelope_text_responses(self):
        env, _ = build_responses_envelope(make_row(endpoint="/v1/responses"), AnswerPayload(text="hi"))
        assert envelope_text(env.model_dump()) == "hi"

    def test_envelope_text_unknown(self):
        assert envelope_text({"object": "other"}) == ""

    def test_is_tool_call_envelope(self):
        text_env, _ = build_chat_response(make_row(), AnswerPayload(text="x"))
        tool_env, _ = build_chat_response(make_row(), AnswerPayload(is_tool_call=True, tool_name="f"))
        assert is_tool_call_envelope(text_env.model_dump()) is False
        assert is_tool_call_envelope(tool_env.model_dump()) is True

    def test_tool_call_parts(self):
        tool_env, _ = build_chat_response(
            make_row(), AnswerPayload(is_tool_call=True, tool_name="get_weather", tool_arguments='{"c":1}')
        )
        parts = tool_call_parts(tool_env.model_dump())
        assert parts is not None
        call_id, name, args = parts
        assert call_id.startswith("call_")
        assert name == "get_weather"
        assert json.loads(args) == {"c": 1}

    def test_tool_call_parts_none_for_text(self):
        text_env, _ = build_chat_response(make_row(), AnswerPayload(text="x"))
        assert tool_call_parts(text_env.model_dump()) is None
