"""POST /v1/responses — the OpenAI Responses API human-in-the-loop endpoint.

Same flow as chat completions; parses the Responses request shape, queues it,
and serves a Responses envelope (non-streaming) or Responses SSE events.
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
    openai_error_response,
    park_and_wait,
    parsed_body_or_error,
)
from ..errors import OpenAIErrorHeld
from ..models import ResponsesRequest, validation_error_payload
from ..summary import summarize_responses

router = APIRouter(tags=["responses"])


@router.post("/responses")
async def create_response(
    request: Request,
    state: AppState = Depends(get_state),
) -> JSONResponse:
    raw = await parsed_body_or_error(request)
    parsed, parse_error = _parse_responses(raw)
    if parse_error is not None:
        return parse_error

    settings = state.settings
    stream = bool(parsed.stream)

    row = build_request_row(
        endpoint="/v1/responses",
        model=parsed.model,
        body_raw=raw,
        parsed_summary=summarize_responses(raw),
        stream_mode=_stream_mode(state, raw),
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
        return JSONResponse(response_row.body_json)

    # Streaming: live word-chunk deltas, or once-mode single delta.
    from fastapi.responses import StreamingResponse

    from ..live_stream import stream_responses_live

    delay_s = settings.stream_chunk_delay_ms / 1000.0

    async def live_gen():
        async for frame in stream_responses_live(
            state.queue,
            row,
            timeout_s=settings.request_timeout_s or None,
            delay_s=delay_s,
        ):
            yield frame

    return StreamingResponse(live_gen(), media_type="text/event-stream")


def _stream_mode(state: AppState, raw: dict[str, Any]) -> str:
    req_mode = raw.get("stream_mode")
    if req_mode in {"word-chunk", "once"}:
        return req_mode
    from ..catalog import load_ui_settings

    ui = load_ui_settings(state.db, state.settings)
    mode = ui.get("stream_mode") or state.settings.stream_mode
    return mode if mode in {"word-chunk", "once"} else "word-chunk"


def _parse_responses(
    raw: dict[str, Any],
) -> tuple[ResponsesRequest | None, JSONResponse | None]:
    """Validate a responses body; return (parsed, None) or (None, error)."""
    try:
        parsed = ResponsesRequest.model_validate(raw)
        parsed.parsed_input_items()
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
