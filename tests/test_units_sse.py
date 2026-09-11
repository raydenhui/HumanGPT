"""Unit tests for humangpt/sse.py — SSE framing + chunk/event assembly."""

from __future__ import annotations

import json

from humangpt.models import new_chat_completion, new_responses_envelope
from humangpt.sse import (
    DONE,
    chat_chunks,
    chat_tool_call_chunks,
    format_sse,
    format_sse_event,
    responses_chunks,
    responses_events,
    split_text_into_chunks,
)


def chat_completion(text="hello world", finish="stop"):
    return new_chat_completion(
        model="human-gpt",
        message={"role": "assistant", "content": text},
        finish_reason=finish,
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        request_id="req_abcdefgh",
    )


def responses_envelope(text="hello world"):
    return new_responses_envelope(
        model="human-gpt",
        outputs=[
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        request_id="req_abcdefgh",
    )


def parse_frames(frames):
    """Split a list of SSE strings into ('data'|'event', payload) tuples."""
    out = []
    for frame in frames:
        event = None
        data = None
        for line in frame.strip().split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        out.append((event, data))
    return out


class TestFraming:
    def test_format_sse(self):
        assert format_sse({"a": 1}) == 'data: {"a":1}\n\n'

    def test_format_sse_event(self):
        assert format_sse_event("response.created", {"a": 1}) == 'event: response.created\ndata: {"a":1}\n\n'

    def test_done_sentinel(self):
        assert DONE == "data: [DONE]\n\n"


class TestSplitTextIntoChunks:
    def test_word_chunk_reconstructs(self):
        text = "Hello world, this is streamed."
        assert "".join(split_text_into_chunks(text, "word-chunk")) == text

    def test_once_is_single(self):
        assert split_text_into_chunks("a b c", "once") == ["a b c"]

    def test_empty(self):
        assert split_text_into_chunks("", "word-chunk") == [""]

    def test_multiline_preserved(self):
        text = "line one\nline two\ttabbed"
        assert "".join(split_text_into_chunks(text, "word-chunk")) == text


class TestChatChunks:
    def test_sequence_and_reconstruction(self):
        frames = list(chat_chunks(chat_completion(), chunks=["hello", " world"], include_usage=False))
        assert frames[-1] == DONE
        parsed = parse_frames(frames[:-1])
        assert all(p[0] is None for p in parsed)  # chat chunks have no event name
        payloads = [json.loads(p[1]) for p in parsed]
        assert all(p["object"] == "chat.completion.chunk" for p in payloads)
        # first chunk carries role
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        reconstructed = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
        assert reconstructed == "hello world"
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"

    def test_include_usage(self):
        frames = list(chat_chunks(chat_completion(), chunks=["hi"], include_usage=True))
        final = json.loads(parse_frames(frames[:-1])[-1][1])
        assert final["usage"]["total_tokens"] == 5

    def test_empty_chunks_is_terminal_only(self):
        frames = list(chat_chunks(chat_completion(), chunks=[], include_usage=False))
        payloads = [json.loads(p[1]) for p in parse_frames(frames[:-1])]
        assert len(payloads) == 1
        assert payloads[0]["choices"][0]["finish_reason"] == "stop"
        assert payloads[0]["choices"][0]["delta"] == {}


class TestChatToolCallChunks:
    def test_sequence(self):
        frames = list(
            chat_tool_call_chunks(
                chat_completion(finish="tool_calls"),
                tool_call_id="call_1",
                name="get_weather",
                arguments='{"city":"Paris"}',
                include_usage=False,
            )
        )
        assert frames[-1] == DONE
        payloads = [json.loads(p[1]) for p in parse_frames(frames[:-1])]
        assert payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
        args = "".join(
            p["choices"][0]["delta"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", "")
            for p in payloads
        )
        assert json.loads(args) == {"city": "Paris"}


class TestResponsesEvents:
    def test_prefix_suffix_order(self):
        env = responses_envelope("hi")
        events, item_id, text = responses_events(env)
        names = [n for n, _ in events]
        assert names[:4] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
        assert names[4:] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert item_id.startswith("msg_")
        assert text == "hi"

    def test_responses_chunks_sequence(self):
        frames = list(responses_chunks(responses_envelope("hello world"), chunks=["hello", " world"]))
        parsed = parse_frames(frames)
        names = [p[0] for p in parsed]
        assert names[:4] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
        assert names[-1] == "response.completed"
        deltas = [json.loads(d)["delta"] for n, d in parsed if n == "response.output_text.delta"]
        assert "".join(deltas) == "hello world"
        # done event carries the full text
        done = [json.loads(d) for n, d in parsed if n == "response.output_text.done"][0]
        assert done["text"] == "hello world"
