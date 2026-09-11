"""Shared helpers for endpoint modules.

Everything an endpoint needs to: parse the JSON body into Pydantic models
(rejecting with OpenAI-style errors), park the request in the queue, wake on
answer, and serve the response (JSON or SSE).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from .db import RequestRow, ResponseRow
from .errors import DiscardedError, InterruptedError, OpenAIErrorHeld, RequestTimeoutError
from .models import ChatCompletion, ResponsesEnvelope
from .parking import QueueService
from .respond import envelope_text, is_tool_call_envelope, tool_call_parts
from .sse import (
    DONE,
    chat_chunks,
    chat_tool_call_chunks,
    responses_chunks,
    split_text_into_chunks,
)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


def openai_error_response(exc: OpenAIErrorHeld) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


async def parsed_body_or_error(request: Request) -> dict[str, Any]:
    """Read the raw JSON body, raising an OpenAI-shaped 400 on malformed JSON."""
    try:
        return json.loads(await request.body())
    except Exception:
        raise OpenAIErrorHeld(
            "The request body is not valid JSON",
            status_code=400,
            error_type="invalid_request_error",
            code=None,
        ) from None


def build_request_row(
    *,
    endpoint: str,
    model: str,
    body_raw: dict[str, Any],
    parsed_summary: dict[str, Any],
    stream_mode: str,
    timeout_s: int | None,
    request_id: str | None = None,
    created_at: float | None = None,
) -> RequestRow:
    """Construct (not yet persisted) a new pending request row."""
    now = created_at if created_at is not None else time.time()
    timeout_at = now + timeout_s if timeout_s is not None and timeout_s > 0 else None
    return RequestRow(
        id=request_id or new_request_id(),
        endpoint=endpoint,
        model=model,
        body=json.dumps(body_raw, separators=(",", ":")),
        parsed=json.dumps(parsed_summary, separators=(",", ":")),
        state="pending",
        created_at=now,
        answered_at=None,
        answered_by=None,
        claimed_by=None,
        claimed_at=None,
        interrupted_at=None,
        stream_mode=stream_mode,
        timeout_at=timeout_at,
    )


async def park_and_wait(
    queue: QueueService,
    request_id: str,
    timeout_s: int,
) -> ResponseRow:
    """Park until answered; returns the persisted ResponseRow.

    Raises OpenAIErrorHeld (timeout/interrupt/discard) when the request is
    released without an answer. The error kind comes from the parking "reason"
    (which the releasing side recorded before resolving), so the wake-up path
    never needs to re-read the database — important during shutdown when the
    DB may already be closed.
    """
    result = await queue.park(request_id, timeout_s or 0)
    if result.answer is None:
        if result.reason == "interrupted":
            raise InterruptedError()
        if result.reason == "discarded":
            raise DiscardedError()
        raise RequestTimeoutError(max(timeout_s or 0, 0))

    response_row = queue.db.get_response_for_request(request_id)
    if response_row is None:
        raise OpenAIErrorHeld(
            "Request was answered but no response record exists",
            status_code=500,
            error_type="server_error",
            code="missing_response",
        )
    return response_row


def non_streaming_chat(response_row: ResponseRow) -> JSONResponse:
    """Serve a persisted chat answer as a plain JSON envelope."""
    return JSONResponse(response_row.body_json)


async def streaming_chat(
    response_row: ResponseRow,
    delay_s: float,
    *,
    include_usage: bool = False,
) -> StreamingResponse:
    """Serve a persisted chat answer as SSE (tool-call or text)."""
    envelope = response_row.body_json
    mode = response_row.stream_mode
    completion = ChatCompletion.model_validate(envelope)

    if is_tool_call_envelope(envelope):
        call = tool_call_parts(envelope)

        async def _gen() -> AsyncIterator[str]:
            tool_call_id, name, arguments = (
                (call[0], call[1], call[2]) if call else (f"call_{uuid.uuid4().hex[:16]}", "unknown_tool", "{}")
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
            for frame in frames:
                yield frame
                await _maybe_delay(frame, delay_s)

        return StreamingResponse(_gen(), media_type="text/event-stream")

    chunks = split_text_into_chunks(envelope_text(envelope), mode)

    async def _gen() -> AsyncIterator[str]:
        frames = list(chat_chunks(completion, chunks=chunks, include_usage=include_usage))
        for i, frame in enumerate(frames):
            yield frame
            if i < len(frames) - 2:  # last two frames are finish_reason + [DONE]
                await _maybe_delay(frame, delay_s)

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def streaming_responses(
    response_row: ResponseRow,
    delay_s: float,
) -> StreamingResponse:
    """Serve a persisted Responses-API answer as SSE."""
    envelope = ResponsesEnvelope.model_validate(response_row.body_json)
    mode = response_row.stream_mode
    chunks = split_text_into_chunks(envelope_text(response_row.body_json), mode)

    async def _gen() -> AsyncIterator[str]:
        frames = list(responses_chunks(envelope, chunks=chunks))
        for frame in frames:
            yield frame
            if "output_text.delta" in frame:
                await _maybe_delay(frame, delay_s)

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _maybe_delay(frame: str, delay_s: float) -> None:
    if delay_s <= 0 or DONE in frame:
        return
    import asyncio

    await asyncio.sleep(delay_s)
