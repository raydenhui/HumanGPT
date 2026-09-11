"""Live-typing streaming delivery.

When stream-mode is ``word-chunk``, the operator's answer is streamed to the
parked client *as the operator types*: each completed word is pushed to the
client immediately; the answering UI locks words it has already sent. The
endpoint's async generator must deliver both kinds of content:

- interim events (``{"type": "delta", "delta": "…"}``) pushed by the
  operator-UI as words lock, and
- the final answer (the persisted response row) pushed on ``resolve``.

The streaming generator drains the parking queue with a short poll so it can
alternate between "any deltas yet?" and "still parked?". On the final event it
assembles the terminal SSE frames (finish_reason, usage when requested,
``[DONE]``) using the *already-emitted* delta text as the assistant content.

``once`` mode keeps the "submit, then whole message" behaviour: the full
answer is pushed as a single delta at submit time.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from . import tokens
from .db import RequestRow
from .errors import DiscardedError, InterruptedError, RequestTimeoutError
from .models import ChatCompletion, new_chat_completion, new_responses_envelope
from .parking import QueueService
from .respond import envelope_text, is_tool_call_envelope, tool_call_parts
from .sse import (
    chat_chunks,
    chat_tool_call_chunks,
    format_sse,
    format_sse_event,
    responses_events,
)


async def stream_chat_live(
    queue: QueueService,
    row: RequestRow,
    *,
    timeout_s: int | None,
    include_usage: bool,
    delay_s: float,
) -> AsyncIterator[str]:
    """Serve a live word-chunk (or once-mode) chat stream.

    Iterates the parking stream: each ``delta`` item yields a chunk frame;
    the terminal ``result`` item carries the answer (or an error).
    """
    emitted_text = ""
    result = None
    async for kind, value in queue.iter_results(row.id, timeout_s):
        if kind == "delta":
            delta = value.get("delta", "")
            emitted_text += delta
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            yield _chat_delta_frame(row, delta)
        else:
            result = value
            break

    if result is None:
        return
    if result.answer is None:
        if result.reason == "interrupted":
            raise InterruptedError()
        if result.reason == "discarded":
            raise DiscardedError()
        raise RequestTimeoutError(max(timeout_s or 0, 0))

    persisted = queue.db.get_response_for_request(row.id)
    envelope = persisted.body_json if persisted is not None else _chat_envelope(row, emitted_text)

    # Reconcile: guarantee the client's accumulated live text equals the
    # submitted answer. If the operator/UI didn't push the final tail as a
    # delta (or a direct API caller only streamed part of it), emit the
    # remaining suffix now. Prefix-safe: never re-sends already-emitted text.
    if not is_tool_call_envelope(envelope):
        full_text = envelope_text(envelope)
        remainder = _missing_suffix(emitted_text, full_text)
        if remainder:
            yield _chat_delta_frame(row, remainder)
            emitted_text += remainder

    completion = ChatCompletion.model_validate(envelope)
    if is_tool_call_envelope(envelope):
        call = tool_call_parts(envelope)
        tool_call_id, name, arguments = (
            (call[0], call[1], call[2])
            if call
            else (f"call_{uuid.uuid4().hex[:16]}", "unknown_tool", "{}")
        )
        frames = list(
            chat_tool_call_chunks(
                completion,
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
                include_usage=include_usage,
            )
        )
    else:
        # deltas were already emitted; terminal frames only (finish + usage + DONE)
        frames = list(chat_chunks(completion, chunks=[], include_usage=include_usage))
    for frame in frames:
        yield frame


async def stream_responses_live(
    queue: QueueService,
    row: RequestRow,
    *,
    timeout_s: int | None,
    delay_s: float,
) -> AsyncIterator[str]:
    """Serve a live word-chunk (or once-mode) Responses-API stream.

    Emits the canonical event order: created → in_progress → item.added →
    content_part.added → (output_text.delta per operator word) →
    output_text.done → content_part.done → output_item.done → completed.

    The prefix is sent when the first delta arrives; the suffix on resolve.
    """
    from .models import ResponsesEnvelope

    # Pre-built prefix (created → content_part.added) with a fixed item id so
    # deltas and suffix reference the same message.
    prefix_env = ResponsesEnvelope.model_validate(_responses_envelope(row, ""))
    prefix_events, item_id, _ = responses_events(prefix_env)
    prefix = prefix_events[:4]

    result = None
    prefix_sent = False
    emitted_text = ""

    async for kind, value in queue.iter_results(row.id, timeout_s):
        if kind == "delta":
            if not prefix_sent:
                for name, payload in prefix:
                    yield format_sse_event(name, payload)
                prefix_sent = True
            delta = value.get("delta", "")
            emitted_text += delta
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            yield _responses_delta_frame(item_id, delta)
        else:
            result = value
            break

    if result is None:
        return
    if result.answer is None:
        if result.reason == "interrupted":
            raise InterruptedError()
        if result.reason == "discarded":
            raise DiscardedError()
        raise RequestTimeoutError(max(timeout_s or 0, 0))

    # The operator's persisted response row is authoritative when present
    # (covers tool-call and once-mode envelopes).
    persisted = queue.db.get_response_for_request(row.id)
    if persisted is not None:
        final = persisted.body_json
    else:
        final = _responses_envelope(row, emitted_text)
    final_env = ResponsesEnvelope.model_validate(final)
    text = final_env.output[0]["content"][0]["text"]

    if not prefix_sent:
        # No live deltas at all (once-mode with empty text edge, or the
        # operator answered without typing): emit the full sequence now.
        events, _, _ = responses_events(final_env)
        for name, payload in events[:4]:
            yield format_sse_event(name, payload)
        if text:
            yield _responses_delta_frame(item_id, text)
        for name, payload in events[4:]:
            yield format_sse_event(name, payload)
        yield format_sse_event("response.completed", {"type": "response.completed", "response": final_env.model_dump()})
        return

    # Reconcile: emit any not-yet-streamed suffix so the client's accumulated
    # text matches the submitted answer (prefix-safe; no duplicates).
    remainder = _missing_suffix(emitted_text, text)
    if remainder:
        yield _responses_delta_frame(item_id, remainder)

    # Live deltas were already streamed; emit the suffix + completed.
    events, _, _ = responses_events(final_env)
    for name, payload in events[4:]:
        yield format_sse_event(name, payload)
    yield format_sse_event("response.completed", {"type": "response.completed", "response": final_env.model_dump()})


def _missing_suffix(emitted: str, full: str) -> str:
    """The part of ``full`` not yet streamed, assuming ``emitted`` is a prefix.

    Returns "" when nothing is missing or when the texts have diverged (in
    which case we don't re-send anything and let the envelope be authoritative).
    """
    if emitted == full:
        return ""
    if full.startswith(emitted):
        return full[len(emitted):]
    if emitted == "":
        return full
    return ""


# ---------------------------------------------------------------- builders

def _chat_delta_frame(row: RequestRow, delta: str) -> str:
    payload = {
        "id": f"chatcmpl-human-{row.id[-8:]}",
        "object": "chat.completion.chunk",
        "created": int(row.created_at),
        "model": row.model,
        "system_fingerprint": "humangpt_fp_000000",
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }
    return format_sse(payload)


def _responses_delta_frame(item_id: str, delta: str) -> str:
    payload = {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }
    return f"event: response.output_text.delta\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chat_envelope(row: RequestRow, text: str) -> dict:
    prompt = int(row.parsed_json.get("prompt_tokens", 1) or 1)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": tokens.estimate_completion_tokens(text),
        "total_tokens": prompt + tokens.estimate_completion_tokens(text),
    }
    message = {"role": "assistant", "content": text}
    return new_chat_completion(
        model=row.model, message=message, finish_reason="stop", usage=usage, request_id=row.id
    ).model_dump()


def _responses_envelope(row: RequestRow, text: str) -> dict:
    prompt = int(row.parsed_json.get("prompt_tokens", 1) or 1)
    usage = {
        "input_tokens": prompt,
        "output_tokens": tokens.estimate_completion_tokens(text),
        "total_tokens": prompt + tokens.estimate_completion_tokens(text),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    outputs = [
        {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    ]
    return new_responses_envelope(
        model=row.model, outputs=outputs, usage=usage, request_id=row.id
    ).model_dump()
