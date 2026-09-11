"""Server-Sent-Events (SSE) encoding for OpenAI-compatible streaming.

Two wire protocols, both consumed by the official SDKs:

- Chat Completions stream (``POST /v1/chat/completions`` with ``stream:true``):
  each event is ``data: {json}\\n\\n``; the sequence is

  ``chunk with role/content`` -> per-token ``content`` deltas ->
  ``finish_reason`` chunk -> (``usage`` chunk when
  ``stream_options.include_usage``) -> ``data: [DONE]``.

  Tool-call answers stream the ``delta.tool_calls`` sequence (id + name first,
  then argument fragments, then ``finish_reason="tool_calls"``).

- Responses API stream (``POST /v1/responses`` with ``stream:true``): each event
  is ``event: <type>\\ndata: {json}\\n\\n``; the sequence is

  response.created -> response.in_progress -> response.output_item.added ->
  response.content_part.added -> response.output_text.delta (per chunk) ->
  response.output_text.done -> response.content_part.done ->
  response.output_item.done -> response.completed.

Chunking: the operator's full message is captured first, then replayed. Two
modes (``MOCK_STREAM_MODE`` / per-answer selection in the UI):

- ``word-chunk`` (default): split on whitespace, keeping the separators, so the
  concatenation ``"".join(deltas)`` reproduces the original text byte-for-byte.
- ``once``: the whole message is emitted as a single content delta.

``format_sse`` / ``format_sse_event`` are pure functions so the same bytes can
be served by FastAPI's StreamingResponse or a future standalone adapter.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from typing import Any

from .models import ChatCompletion, ResponsesEnvelope

StreamMode = str  # "word-chunk" | "once"


def format_sse(payload: dict[str, Any]) -> str:
    """Serialise one ``data: {...}\\n\\n`` frame (chat completions style)."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def format_sse_event(event: str, payload: dict[str, Any]) -> str:
    """Serialise one named ``event: X\\ndata: {...}\\n\\n`` frame (responses style)."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


DONE = "data: [DONE]\n\n"


def split_text_into_chunks(text: str, mode: StreamMode) -> list[str]:
    """Split a human answer into streamable deltas that reconstruct ``text``.

    ``"".join(chunks)`` always equals ``text``; interior chunks carry the
    whitespace that preceded them in the source, so newlines, tabs and doubled
    spaces survive the round-trip.
    """
    if mode == "once":
        return [text]
    parts = re.split(r"(\s+)", text)
    if not parts:
        return [""]
    chunks = [parts[0]]
    for i in range(1, len(parts), 2):
        separator = parts[i]
        word = parts[i + 1] if i + 1 < len(parts) else ""
        chunks.append(separator + word)
    return chunks


def _base_payload(
    completion_id: str,
    created: int,
    model: str,
    system_fingerprint: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
    }


def chat_chunks(
    completion: ChatCompletion,
    *,
    chunks: list[str],
    include_usage: bool,
) -> Iterator[str]:
    """Yield chat-completions SSE frames for a single text answer.

    When ``chunks`` is empty (live-typing mode: deltas were already emitted),
    only the terminal chunk (+usage) + ``[DONE]`` are produced.
    """
    choice = completion.choices[0]
    base = _base_payload(completion.id, completion.created, completion.model, completion.system_fingerprint)

    for i, delta_text in enumerate(chunks):
        payload = dict(base)
        if i == 0:
            payload["choices"] = [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": None,
                }
            ]
        else:
            payload["choices"] = [
                {
                    "index": 0,
                    "delta": {"content": delta_text},
                    "finish_reason": None,
                }
            ]
        yield format_sse(payload)

    final = dict(base)
    final["choices"] = [{"index": 0, "delta": {}, "finish_reason": choice.finish_reason}]
    if include_usage:
        final["usage"] = completion.usage.model_dump()
    yield format_sse(final)
    yield DONE


def chat_tool_call_chunks(
    completion: ChatCompletion,
    *,
    tool_call_id: str,
    name: str,
    arguments: str,
    include_usage: bool,
) -> Iterator[str]:
    """Yield chat-completions SSE frames for a tool_call answer.

    Emits the standard function-call sequence: first chunk carries
    ``delta.tool_calls[0]`` with id/type/function.name and empty arguments,
    then one chunk appends the full arguments JSON, then a
    ``finish_reason="tool_calls"`` chunk and ``[DONE]``.
    """
    base = _base_payload(completion.id, completion.created, completion.model, completion.system_fingerprint)

    first = dict(base)
    first["choices"] = [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ],
            },
            "finish_reason": None,
        }
    ]
    yield format_sse(first)

    arg_chunk = dict(base)
    arg_chunk["choices"] = [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {"index": 0, "id": tool_call_id, "type": "function", "function": {"arguments": arguments}}
                ]
            },
            "finish_reason": None,
        }
    ]
    yield format_sse(arg_chunk)

    final = dict(base)
    final["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    if include_usage:
        final["usage"] = completion.usage.model_dump()
    yield format_sse(final)
    yield DONE


def _response_base(envelope: ResponsesEnvelope) -> dict[str, Any]:
    return {
        "id": envelope.id,
        "object": envelope.object,
        "created_at": envelope.created_at,
        "status": envelope.status,
        "model": envelope.model,
        "output": envelope.output,
        "parallel_tool_calls": envelope.parallel_tool_calls,
        "tool_choice": envelope.tool_choice,
        "usage": envelope.usage,
        "error": None,
        "incomplete_details": None,
    }


def responses_chunks(
    envelope: ResponsesEnvelope,
    *,
    chunks: list[str],
) -> Iterator[str]:
    """Yield Responses-API SSE frames for a single text output item.

    ``envelope.output[0]`` must be a ``message`` output item whose first content
    part is ``{"type": "output_text", "text": ...}``.
    """
    item_id = f"msg_{uuid.uuid4().hex[:8]}"
    text = envelope.output[0]["content"][0]["text"]

    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"type": "response.created", "response": _response_base(envelope)}),
        ("response.in_progress", {"type": "response.in_progress", "response": _response_base(envelope)}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
    ]

    for delta_text in chunks:
        events.append(
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta_text,
                },
            )
        )

    events.extend(
        [
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            ),
            (
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text},
                },
            ),
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    },
                },
            ),
        ]
    )

    for event, payload in events:
        yield format_sse_event(event, payload)

    yield format_sse_event("response.completed", {"type": "response.completed", "response": _response_base(envelope)})


def responses_events(
    envelope: ResponsesEnvelope,
) -> tuple[list[tuple[str, dict[str, Any]]], str, str]:
    """Return the (event-name, payload) list for a Responses stream.

    Separated from the wire framing so live-streaming can emit the prefix,
    then interleave operator-typed deltas, then the suffix, all with identical
    event names/order — without string-matching frames.

    Returns ``(events, item_id, text)`` where ``events`` contains the
    prefix + suffix (the ``output_text.delta`` events are *not* included —
    the caller injects those as they stream).
    """
    item_id = f"msg_{uuid.uuid4().hex[:8]}"
    text = envelope.output[0]["content"][0]["text"]
    base = _response_base(envelope)

    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"type": "response.created", "response": base}),
        ("response.in_progress", {"type": "response.in_progress", "response": base}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant", "content": []},
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text},
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
        ),
    ]
    return events, item_id, text
