"""Unit tests for humangpt/models.py — request parsing / envelope construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from humangpt.models import (
    ChatCompletionsRequest,
    ContentPart,
    ResponsesRequest,
    new_chat_completion,
    new_responses_envelope,
    parse_content_parts,
    validation_error_payload,
)


class TestParseContentParts:
    def test_none_is_empty(self):
        assert parse_content_parts(None) == []

    def test_string_becomes_text_part(self):
        parts = parse_content_parts("hello")
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert parts[0].text == "hello"

    def test_list_of_parts(self):
        parts = parse_content_parts(
            [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}},
            ]
        )
        assert len(parts) == 2
        assert parts[1].type == "image_url"
        assert parts[1].image_url_value == "https://x/y.png"
        assert parts[1].image_detail == "low"

    def test_image_url_as_bare_string(self):
        parts = parse_content_parts([{"type": "image_url", "image_url": "https://x/y.png"}])
        assert parts[0].image_url_value == "https://x/y.png"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            parse_content_parts(123)


class TestChatCompletionsRequest:
    def test_minimal_valid(self):
        req = ChatCompletionsRequest.model_validate(
            {"model": "human-gpt", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert req.model == "human-gpt"
        assert req.parsed_messages()[0]["role"] == "user"

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionsRequest.model_validate({"messages": []})

    def test_unknown_fields_tolerated(self):
        req = ChatCompletionsRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "totally_unknown": 1}
        )
        assert req.model == "m"

    def test_bad_role_rejected(self):
        req = ChatCompletionsRequest.model_validate(
            {"model": "m", "messages": [{"role": "wizard", "content": "hi"}]}
        )
        with pytest.raises(ValueError):
            req.parsed_messages()

    def test_user_without_content_rejected(self):
        req = ChatCompletionsRequest.model_validate({"model": "m", "messages": [{"role": "user"}]})
        with pytest.raises(ValueError):
            req.parsed_messages()

    def test_include_usage(self):
        assert ChatCompletionsRequest.model_validate(
            {"model": "m", "messages": [], "stream_options": {"include_usage": True}}
        ).include_usage
        assert not ChatCompletionsRequest.model_validate(
            {"model": "m", "messages": [], "stream_options": {}}
        ).include_usage
        assert not ChatCompletionsRequest.model_validate({"model": "m", "messages": []}).include_usage


class TestResponsesRequest:
    def test_string_input(self):
        req = ResponsesRequest.model_validate({"model": "m", "input": "hello"})
        assert req.parsed_input_items() == [{"type": "message", "role": "user", "content": "hello"}]

    def test_array_input(self):
        req = ResponsesRequest.model_validate(
            {"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}]}
        )
        items = req.parsed_input_items()
        assert items[0]["role"] == "user"

    def test_array_with_bare_string_entries(self):
        req = ResponsesRequest.model_validate({"model": "m", "input": ["hi", "there"]})
        items = req.parsed_input_items()
        assert len(items) == 2
        assert items[0]["content"] == "hi"

    def test_missing_input_raises(self):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate({"model": "m"})


class TestEnvelopeFactories:
    def test_new_chat_completion(self):
        env = new_chat_completion(
            model="human-gpt",
            message={"role": "assistant", "content": "hi"},
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            request_id="req_abcdefgh",
        )
        assert env.id == "chatcmpl-human-req_abcd"
        assert env.object == "chat.completion"
        assert env.choices[0].message["content"] == "hi"

    def test_new_responses_envelope(self):
        env = new_responses_envelope(
            model="human-gpt",
            outputs=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}],
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            request_id="req_abcdefgh",
        )
        assert env.id == "resp-human-req_abcd"
        assert env.object == "response"
        assert env.usage["total_tokens"] == 2


class TestValidationErrorPayload:
    def test_shape(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionsRequest.model_validate({"messages": []})
        status, payload = validation_error_payload(excinfo.value)
        assert status == 400
        assert payload["error"]["type"] == "invalid_request_error"
        assert "message" in payload["error"]


class TestContentPartAccessors:
    def test_text_part(self):
        part = ContentPart(type="text", text="hi")
        assert part.kind == "text"
        assert part.image_url_value is None

    def test_image_part_with_string(self):
        part = ContentPart(type="image_url", image_url="https://x/y.png")
        assert part.image_url_value == "https://x/y.png"
