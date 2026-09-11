"""POST /v1/chat/completions — the chat-completions human-in-the-loop endpoint.

Flow: parse & validate the body (OpenAI-shaped errors on failure) -> persist a
pending request row -> park the HTTP request in the queue -> when the human
answers, serve the persisted response (JSON or SSE).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..container import AppState
from ..deps import get_state
from ..endpoint_utils import (
    build_request_row,
    non_streaming_chat,
    openai_error_response,
    park_and_wait,
    parsed_body_or_error,
)
from ..errors import OpenAIErrorHeld
from ..models import ChatCompletionsRequest, validation_error_payload
from ..summary import summarize_chat

router = APIRouter(tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    state: AppState = Depends(get_state),
) -> JSONResponse:
    raw = await parsed_body_or_error(request)
    parsed, parse_error = _parse_chat(raw)
    if parse_error is not None:
        return parse_error

    settings = state.settings
    stream = bool(parsed.stream)
    stream_mode = _effective_stream_mode(state, raw)

    row = build_request_row(
        endpoint="/v1/chat/completions",
        model=parsed.model,
        body_raw=raw,
        parsed_summary=summarize_chat(raw),
        stream_mode=stream_mode,
        timeout_s=settings.request_timeout_s or None,
    )
    state.db.create_request(row)
    state.hub.publish({"type": "request-changed", "id": row.id, "state": "pending"})

    if not stream:
        response_row = await park_and_wait(
            state.queue,
            row.id,
            settings.request_timeout_s or None,
        )
        return non_streaming_chat(response_row)

    # Streaming (live word-chunk or submit-then-delta for "once"):
    # stream_chat_live drains interim deltas and finishes with terminal frames.
    from fastapi.responses import StreamingResponse

    from ..live_stream import stream_chat_live

    delay_s = settings.stream_chunk_delay_ms / 1000.0

    async def live_gen():
        async for frame in stream_chat_live(
            state.queue,
            row,
            timeout_s=settings.request_timeout_s or None,
            include_usage=parsed.include_usage,
            delay_s=delay_s,
        ):
            yield frame

    return StreamingResponse(live_gen(), media_type="text/event-stream")


def _effective_stream_mode(state: AppState, raw: dict[str, Any]) -> str:
    """Resolve the stream mode for the request.

    The operator's global setting (Settings panel) governs; a per-request
    ``stream_mode`` field (used by the SDK/tests) can override.
    """
    req_mode = raw.get("stream_mode")
    if req_mode in {"word-chunk", "once"}:
        return req_mode
    from ..catalog import load_ui_settings

    ui = load_ui_settings(state.db, state.settings)
    mode = ui.get("stream_mode") or state.settings.stream_mode
    return mode if mode in {"word-chunk", "once"} else "word-chunk"


def _parse_chat(
    raw: dict[str, Any],
) -> tuple[ChatCompletionsRequest | None, JSONResponse | None]:
    """Validate a chat-completions body; return (parsed, None) or (None, error)."""
    try:
        parsed = ChatCompletionsRequest.model_validate(raw)
        parsed.parsed_messages()
        return parsed, None
    except ValidationError as exc:
        status, payload = validation_error_payload(exc)
        return None, openai_error_response(_from_payload(status, payload))
    except ValueError as exc:
        return None, openai_error_response(
            OpenAIErrorHeld(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
            )
        )


def _from_payload(status: int, payload: dict[str, Any]) -> OpenAIErrorHeld:
    err = payload["error"]
    return OpenAIErrorHeld(
        message=err["message"],
        status_code=status,
        error_type=err["type"],
        param=err.get("param"),
        code=err.get("code"),
    )
