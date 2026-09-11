"""Unit tests for humangpt/summary.py — the operator-facing parsed summary."""

from __future__ import annotations

from humangpt.summary import summarize_chat, summarize_responses


class TestSummarizeChat:
    def test_basic_shape(self):
        body = {
            "model": "human-gpt",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello there"},
            ],
            "temperature": 0.5,
        }
        s = summarize_chat(body)
        assert s["kind"] == "chat"
        assert s["endpoint"] == "/v1/chat/completions"
        assert s["prompt_tokens"] >= 1
        assert s["role_counts"] == {"system": 1, "user": 1}
        assert s["first_user_text"] == "sys"  # first system/user text
        assert s["has_images"] is False
        assert s["message_count"] == 2
        assert s["params"]["temperature"] == 0.5

    def test_images_detected(self):
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]}
        s = summarize_chat(body)
        assert s["has_images"] is True

    def test_tool_names(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        s = summarize_chat(body)
        assert s["tool_names"] == ["get_weather"]

    def test_snippet_truncated(self):
        body = {"messages": [{"role": "user", "content": "x" * 500}]}
        s = summarize_chat(body)
        assert len(s["snippet"]) <= 120

    def test_empty_messages(self):
        s = summarize_chat({})
        assert s["message_count"] == 0
        assert s["first_user_text"] == ""


class TestSummarizeResponses:
    def test_string_input(self):
        s = summarize_responses({"model": "human-gpt", "input": "a string input", "instructions": "be nice"})
        assert s["kind"] == "responses"
        assert s["endpoint"] == "/v1/responses"
        assert s["message_count"] == 1
        assert s["first_user_text"] == "a string input"
        assert s["has_images"] is False

    def test_item_array_with_image(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA"},
                    ],
                }
            ]
        }
        s = summarize_responses(body)
        assert s["has_images"] is True
        assert s["message_count"] == 1
        assert s["first_user_text"] == "what is this"

    def test_instructions_in_prompt_tokens(self):
        with_i = summarize_responses({"input": "hi", "instructions": "a long set of instructions " * 20})
        without = summarize_responses({"input": "hi"})
        assert with_i["prompt_tokens"] > without["prompt_tokens"]

    def test_role_counts(self):
        body = {
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "message", "role": "assistant", "content": "hello"},
            ]
        }
        s = summarize_responses(body)
        assert s["role_counts"]["user"] == 1
        assert s["role_counts"]["assistant"] == 1
