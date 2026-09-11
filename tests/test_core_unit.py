"""Unit tests for the pure, I/O-free core logic in humangpt/core.py.

These run without any network/DB/event-loop and cover the important functions:
state transitions, stream-mode sanitization + word chunking, token heuristics,
answer-envelope construction, and the live buffer assembly.
"""

from __future__ import annotations

import pytest

from humangpt.core import (
    ANSWERED,
    DISCARDED,
    INTERRUPTED,
    ONCE,
    PENDING,
    STREAMED,
    TIMED_OUT,
    WORD_CHUNK,
    InvalidTransition,
    LiveBuffer,
    assemble_text_from_deltas,
    build_chat_usage,
    build_response_envelope,
    can_transition,
    estimate_completion_tokens,
    lock_word,
    sanitize_stream_mode,
    split_into_deltas,
    terminal_state,
    transition,
)


class TestStateMachine:
    def test_pending_can_transition_to_all_results(self):
        for target in (ANSWERED, STREAMED, TIMED_OUT, INTERRUPTED, DISCARDED):
            assert can_transition(PENDING, target)

    def test_answered_can_return_to_pending(self):
        assert can_transition(ANSWERED, PENDING)
        assert can_transition(STREAMED, PENDING)
        assert can_transition(TIMED_OUT, PENDING)
        assert can_transition(INTERRUPTED, PENDING)
        assert can_transition(DISCARDED, PENDING)

    def test_terminal_states(self):
        for s in (ANSWERED, STREAMED, TIMED_OUT, INTERRUPTED, DISCARDED):
            assert terminal_state(s)
        assert not terminal_state(PENDING)

    def test_illegal_transition_raises(self):
        with pytest.raises(InvalidTransition):
            transition(PENDING, PENDING, "req1")

    def test_transition_returns_target(self):
        assert transition(PENDING, ANSWERED, "req1") == ANSWERED

    def test_pending_to_pending_illegal_in_table(self):
        assert not can_transition(PENDING, PENDING)
        assert not can_transition(ANSWERED, DISCARDED)


class TestStreamModeAndChunking:
    def test_sanitize_accepts_known(self):
        assert sanitize_stream_mode("word-chunk") == WORD_CHUNK
        assert sanitize_stream_mode("once") == ONCE

    def test_sanitize_falls_back_to_default(self):
        assert sanitize_stream_mode("bogus") == WORD_CHUNK
        assert sanitize_stream_mode(None) == WORD_CHUNK
        assert sanitize_stream_mode("bogus", default="once") == ONCE

    def test_split_once_is_single_delta(self):
        assert split_into_deltas("hello world", ONCE) == ["hello world"]

    def test_split_word_chunk_reconstructs_exactly(self):
        text = "Hello world, this is streamed."
        deltas = split_into_deltas(text, WORD_CHUNK)
        assert "".join(deltas) == text
        assert len(deltas) >= 4  # at least word-shaped

    def test_split_word_chunk_multiline(self):
        text = "line one\nline two\ttabbed"
        deltas = split_into_deltas(text, WORD_CHUNK)
        assert "".join(deltas) == text

    def test_split_empty_text(self):
        assert split_into_deltas("", WORD_CHUNK) == [""]

    def test_lock_word_leaves_tail(self):
        locked, tail = lock_word("Hello ")
        assert locked == "Hello"
        assert tail == " "

    def test_lock_word_unlatch_without_separator(self):
        locked, tail = lock_word("Hello")
        assert locked == ""
        assert tail == "Hello"

    def test_lock_word_tab_separator(self):
        locked, tail = lock_word("Hello\t")
        assert locked == "Hello"
        assert tail == "\t"

    def test_lock_word_rides_leading_separator(self):
        # the SPA's tail after locking "1" is " "; typing "2 " gives " 2 "
        locked, tail = lock_word(" 2 ")
        assert locked == " 2"
        assert tail == " "

    def test_lock_word_sequence_no_trailing_space(self):
        """Simulate typing "1 2 3" char-by-char (mirrors ui/src/wordlock.js).

        "3" (no trailing space) stays in the tail until finish; finish trims
        and appends it.
        """
        locked: list[str] = []
        tail = ""
        for ch in "1 2 3":
            val = tail + ch
            delta, tail = lock_word(val)
            if delta:
                locked.append(delta)
        assert locked == ["1", " 2"]
        # finish(): trim trailing whitespace of the tail, push the rest.
        # The tail is " 3" (leading separator + final word) — cleaned is " 3".
        cleaned = tail.rstrip()
        assert cleaned == " 3"
        locked.append(cleaned)
        assert "".join(locked) == "1 2 3"
        assert "".join(locked).endswith(" ") is False

    def test_lock_word_sequence_one_space_per_separator(self):
        """Typing "1 2 3 4 5" uses each space exactly once (no double-space)."""
        locked: list[str] = []
        tail = ""
        for ch in "1 2 3 4 5":
            val = tail + ch
            delta, tail = lock_word(val)
            if delta:
                locked.append(delta)
        assert locked == ["1", " 2", " 3", " 4"]
        assert tail == " 5"
        assert "".join(locked).endswith(" ") is False
        # finish
        locked.append(tail.rstrip())
        assert "".join(locked) == "1 2 3 4 5"


class TestTokens:
    def test_completion_tokens_positive(self):
        assert estimate_completion_tokens("hi") >= 1
        assert estimate_completion_tokens("") == 1

    def test_completion_tokens_scales_with_bytes(self):
        short_t = estimate_completion_tokens("a" * 10)
        long_t = estimate_completion_tokens("a" * 1000)
        assert long_t > short_t


class TestEnvelopeBuilding:
    def test_text_message(self):
        env = build_response_envelope(
            request_id="req_abc123", model="human-gpt", prompt_tokens=10, text="hi there"
        )
        assert env["object"] == "chat.completion"
        assert env["model"] == "human-gpt"
        assert env["id"].startswith("chatcmpl-human-")
        choice = env["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "hi there"
        assert env["usage"]["prompt_tokens"] == 10
        assert env["usage"]["total_tokens"] == 10 + env["usage"]["completion_tokens"]

    def test_tool_message(self):
        env = build_response_envelope(
            request_id="req_xyz",
            model="human-gpt",
            prompt_tokens=5,
            is_tool_call=True,
            tool_name="get_weather",
            tool_arguments='{"city":"Paris"}',
        )
        choice = env["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert call["function"]["arguments"] == '{"city":"Paris"}'
        assert call["type"] == "function"

    def test_usage_totals(self):
        usage = build_chat_usage(12, "hi")
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["completion_tokens"] >= 1


class TestLiveBuffer:
    def test_append_and_reconstruct(self):
        b = LiveBuffer()
        b.append_word("Hello ")
        b.append_word("world ")
        assert b.text == "Hello world "
        b.finish_with_tail("!")
        assert b.text == "Hello world !"

    def test_empty(self):
        assert LiveBuffer().text == ""

    def test_deltas_assemble(self):
        assert assemble_text_from_deltas(["a", " b"]) == "a b"
