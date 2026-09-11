"""Unit tests for humangpt/tokens.py — the documented usage heuristic."""

from __future__ import annotations

from humangpt.tokens import (
    encode_to_tokens,
    estimate_chat_prompt_tokens,
    estimate_completion_tokens,
    estimate_responses_prompt_tokens,
)


class TestEncodeToTokens:
    def test_min_one(self):
        assert encode_to_tokens("") == 1
        assert encode_to_tokens("a") >= 1

    def test_scales_with_bytes(self):
        assert encode_to_tokens("a" * 4000) > encode_to_tokens("a" * 40)

    def test_utf8_bytes_counted(self):
        # multi-byte chars should count more tokens than their char length
        assert encode_to_tokens("é" * 100) >= encode_to_tokens("a" * 100)


class TestChatPromptTokens:
    def test_string_content(self):
        tokens = estimate_chat_prompt_tokens([{"role": "user", "content": "hello world"}], None)
        assert tokens >= 1

    def test_text_parts(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert estimate_chat_prompt_tokens(msgs, None) >= 1

    def test_data_uri_image_heuristic(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
            }
        ]
        assert estimate_chat_prompt_tokens(msgs, None) == 765

    def test_http_image_heuristic(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
        assert estimate_chat_prompt_tokens(msgs, None) == 85

    def test_tools_counted(self):
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        assert estimate_chat_prompt_tokens(msgs, tools) > estimate_chat_prompt_tokens(msgs, None)

    def test_instructions_counted(self):
        msgs = [{"role": "user", "content": "hi"}]
        with_instr = estimate_chat_prompt_tokens(msgs, None, "be terse and precise")
        without = estimate_chat_prompt_tokens(msgs, None)
        assert with_instr > without

    def test_tool_calls_counted(self):
        msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}]
        assert estimate_chat_prompt_tokens(msgs, None) >= 1


class TestResponsesPromptTokens:
    def test_text_item(self):
        items = [{"type": "message", "role": "user", "content": "hello"}]
        assert estimate_responses_prompt_tokens(items, None) >= 1

    def test_input_image(self):
        items = [{"type": "message", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]
        assert estimate_responses_prompt_tokens(items, None) == 765

    def test_function_call_output(self):
        items = [{"type": "function_call_output", "output": "42"}]
        assert estimate_responses_prompt_tokens(items, None) >= 1


class TestCompletionTokens:
    def test_positive(self):
        assert estimate_completion_tokens("hello") >= 1
        assert estimate_completion_tokens("") == 1
