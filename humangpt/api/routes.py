"""Operator JSON API (powered by ``humangpt.core`` and the FastAPI app state).

The SPA (React) talks to these /api/* endpoints; the /v1/* OpenAI surface stays
JSON-compatible for typed SDK clients. Logic lives in ``humangpt.core`` (pure)
so it is unit-tested without network/DB.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.requests import Request

from ..catalog import apply_ui_settings, load_catalog, load_ui_settings
from ..container import AppState
from ..core import sanitize_stream_mode, transition
from ..db import RequestRow, ResponseRow
from ..deps import get_state
from ..endpoint_utils import parsed_body_or_error
from ..errors import RequestNotPendingError
from ..respond import AnswerPayload, build_chat_response, build_responses_envelope

router = APIRouter(prefix="/api", tags=["operator-api"])

STATUS_LABELS = {
    "pending": "pending",
    "answered": "answered",
    "streamed": "streamed",
    "timed_out": "timed out",
    "interrupted": "interrupted",
    "discarded": "discarded",
}


def _fmt_request(r: RequestRow) -> dict[str, Any]:
    parsed = r.parsed_json
    body = r.body_json
    return {
        "id": r.id,
        "endpoint": r.endpoint,
        "model": r.model,
        "state": r.state,
        "status": STATUS_LABELS.get(r.state, r.state),
        "created_at": r.created_at,
        "snippet": parsed.get("snippet", ""),
        "has_images": bool(parsed.get("has_images")),
        "tool_names": parsed.get("tool_names", []),
        "claimed_by": r.claimed_by,
        "message_count": parsed.get("message_count", 0),
        "stream": bool(body.get("stream")),
        "answered_by": r.answered_by,
    }


def _stream_default(state: AppState) -> str:
    return sanitize_stream_mode(load_ui_settings(state.db, state.settings)["stream_mode"])


# ---------------------------------------------------------------------------
# Meta (for the header badge + API URL)
# ---------------------------------------------------------------------------


@router.get("/meta")
async def meta(request: Request, state: AppState = Depends(get_state)):
    base_url = (
        f"{request.base_url.scheme}://{request.base_url.netloc}" if request else None
    )
    return {
        "api_base_url": f"{base_url or f'http://{state.settings.host}:{state.settings.port}'}/v1",
        "pending_count": len(state.db.list_requests(states=["pending"], limit=1000)),
        "models": [m["id"] for m in state.models],
    }


# ---------------------------------------------------------------------------
# Queue + request detail
# ---------------------------------------------------------------------------


@router.get("/requests")
async def list_pending(state: AppState = Depends(get_state)):
    rows = state.db.list_requests(states=["pending"], limit=300)
    return {"pending": [_fmt_request(r) for r in reversed(rows)]}


@router.get("/requests/{request_id}")
async def request_detail(request_id: str, state: AppState = Depends(get_state)):
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    return {
        "request": _fmt_request(row),
        "body": row.body_json,
        "parsed": row.parsed_json,
        "response": _response_json(state, row.id),
        "stream_mode": _stream_default(state),
        "templates": [{"name": t.name, "body": t.body} for t in state.db.list_templates()],
    }


def _response_json(state: AppState, request_id: str) -> dict[str, Any] | None:
    resp = state.db.get_response_for_request(request_id)
    return resp.body_json if resp else None


# ---------------------------------------------------------------------------
# Operator actions
# ---------------------------------------------------------------------------


@router.post("/requests/{request_id}/claim")
async def claim(request_id: str, request: Request, state: AppState = Depends(get_state)):
    # The SPA claims with an empty body; accept an optional JSON operator name.
    operator = "anonymous"
    try:
        raw = await request.body()
        if raw:
            import json as _json

            payload = _json.loads(raw)
            operator = (payload.get("operator") or "anonymous").strip() or "anonymous"
    except Exception:
        operator = "anonymous"
    state.db.claim_request(request_id, operator)
    state.hub.publish({"type": "request-changed", "id": request_id, "state": "pending"})
    return {"ok": True}


@router.post("/requests/{request_id}/stream-word")
async def stream_word(request_id: str, request: Request, state: AppState = Depends(get_state)):
    payload = await parsed_body_or_error(request)
    delta = payload.get("delta", "")
    if not isinstance(delta, str):
        return JSONResponse({"error": "delta must be a string"}, status_code=400)
    row = state.db.get_request(request_id)
    if row is None or row.state != "pending":
        return JSONResponse({"error": "not pending"}, status_code=409)
    state.queue.emit(request_id, {"type": "delta", "delta": delta})
    return {"ok": True}


@router.post("/requests/{request_id}/answer")
async def answer(request_id: str, request: Request, state: AppState = Depends(get_state)):
    payload = await parsed_body_or_error(request)
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    if row.state != "pending":
        raise RequestNotPendingError(request_id, row.state)

    text = payload.get("text", "")
    is_tool = bool(payload.get("is_tool_call"))
    tool_name = payload.get("tool_name", "")
    tool_arguments = payload.get("tool_arguments", "")
    stream_mode = sanitize_stream_mode(payload.get("stream_mode"), _stream_default(state))
    operator = payload.get("operator") or "anonymous"

    if is_tool and not tool_name.strip():
        return JSONResponse({"error": "tool_name is required for a tool_call answer"}, status_code=400)
    if is_tool:
        try:
            json.loads(tool_arguments or "{}")
        except json.JSONDecodeError as exc:
            return JSONResponse({"error": f"tool_arguments must be valid JSON: {exc.msg}"}, status_code=400)

    payload_obj = AnswerPayload(
        text=text,
        is_tool_call=is_tool,
        tool_name=tool_name.strip() if is_tool else None,
        tool_arguments=tool_arguments if is_tool else None,
        stream_mode=stream_mode,
    )
    envelope, record = _build_envelope(row, payload_obj)

    now = time.time()
    response_row = ResponseRow(
        id=f"resp_{uuid.uuid4().hex[:16]}",
        request_id=row.id,
        body=json.dumps(envelope, separators=(",", ":")),
        finish_reason="tool_calls" if is_tool else "stop",
        is_tool_call=is_tool,
        tool_name=record["tool_name"],
        tool_arguments=record["tool_arguments"],
        stream_mode=record["stream_mode"],
        created_at=now,
    )
    target = "streamed" if bool(row.body_json.get("stream")) else "answered"
    if not state.db.update_request_state(row.id, target, answered_at=now, answered_by=operator, only_if_pending=True):
        return JSONResponse({"error": "request already answered"}, status_code=409)
    state.db.create_response(response_row)

    # once-mode: push the full answer as a delta so the parked live stream's
    # accumulated text matches the envelope.
    if row.body_json.get("stream") and stream_mode == "once":
        state.queue.emit(request_id, {"type": "delta", "delta": text})

    state.queue.resolve(row.id, answer={"response_id": response_row.id}, reason="answered")
    state.hub.publish({"type": "request-changed", "id": row.id, "state": target})
    return {"ok": True, "request_id": row.id, "state": target}


def _build_envelope(row: RequestRow, payload_obj: AnswerPayload):
    if row.endpoint == "/v1/responses":
        envelope, record = build_responses_envelope(row, payload_obj)
        return envelope.model_dump(), record
    envelope, record = build_chat_response(row, payload_obj)
    return envelope.model_dump(), record


@router.post("/requests/{request_id}/return")
async def return_to_pending(request_id: str, state: AppState = Depends(get_state)):
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    transition(row.state, "pending", request_id)
    state.db.update_request_state(row.id, "pending", answered_at=None, answered_by=None)
    timeout_s = state.settings.request_timeout_s
    state.db._execute(
        "UPDATE requests SET timeout_at = ? WHERE id = ?",
        (time.time() + timeout_s if timeout_s and timeout_s > 0 else None, row.id),
    )
    state.db.commit()
    state.db.unclaim_request(row.id, row.claimed_by or "")
    state.hub.publish({"type": "request-changed", "id": row.id, "state": "pending"})
    return {"ok": True}


@router.post("/requests/{request_id}/timeout")
async def mark_timed_out(request_id: str, state: AppState = Depends(get_state)):
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    if not state.db.update_request_state(row.id, "timed_out", answered_at=time.time(), only_if_pending=True):
        return JSONResponse({"error": "request not pending"}, status_code=409)
    state.queue.resolve(row.id, answer=None, reason="timeout")
    state.hub.publish({"type": "request-changed", "id": row.id, "state": "timed_out"})
    return {"ok": True}


@router.post("/requests/{request_id}/discard")
async def discard(request_id: str, state: AppState = Depends(get_state)):
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    if not state.db.discard_request(row.id):
        return JSONResponse({"error": "request not pending"}, status_code=409)
    state.queue.resolve(row.id, answer=None, reason="discarded")
    state.hub.publish({"type": "request-changed", "id": row.id, "state": "discarded"})
    return {"ok": True}


# ---------------------------------------------------------------------------
# History (for the SPA transcript view)
# ---------------------------------------------------------------------------


@router.get("/history")
async def history_list(state: AppState = Depends(get_state)):
    rows = state.db.list_requests(
        states=["answered", "streamed", "timed_out", "interrupted", "discarded"], limit=500
    )
    return {"rows": [_fmt_request(r) for r in reversed(rows)]}


@router.get("/history/{request_id}")
async def history_detail(request_id: str, state: AppState = Depends(get_state)):
    row = state.db.get_request(request_id)
    if row is None:
        return JSONResponse({"error": "no such request"}, status_code=404)
    return {
        "request": {**_fmt_request(row), "parsed": row.parsed_json},
        "body": row.body_json,
        "response": _response_json(state, row.id),
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@router.get("/templates")
async def list_templates(state: AppState = Depends(get_state)):
    return {"templates": [{"name": t.name, "body": t.body} for t in state.db.list_templates()]}


@router.post("/templates")
async def save_template(request: Request, state: AppState = Depends(get_state)):
    payload = await parsed_body_or_error(request)
    name = (payload.get("name") or "").strip()
    body = payload.get("body") or ""
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    state.db.create_template(name, body)
    state.hub.publish({"type": "templates-changed"})
    return {"ok": True}


@router.post("/templates/{name}/delete")
async def delete_template(name: str, state: AppState = Depends(get_state)):
    state.db.delete_template(name)
    state.hub.publish({"type": "templates-changed"})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Settings + models editor
# ---------------------------------------------------------------------------


@router.get("/settings")
async def get_settings(request: Request, state: AppState = Depends(get_state)):
    ui = load_ui_settings(state.db, state.settings)
    base_url = f"{request.base_url.scheme}://{request.base_url.netloc}"
    return {
        "stream_mode": ui["stream_mode"],
        "stream_chunk_delay_ms": ui["stream_chunk_delay_ms"],
        "api_base_url": f"{base_url}/v1",
        "configured_key": bool(state.settings.openai_api_key),
        "models": state.models,
    }


@router.post("/settings")
async def save_settings(request: Request, state: AppState = Depends(get_state)):
    payload = await parsed_body_or_error(request)
    apply_ui_settings(state.db, {
        "stream_mode": payload.get("stream_mode", ""),
        "stream_chunk_delay_ms": payload.get("stream_chunk_delay_ms", 0),
    })
    state.hub.publish({"type": "settings-changed"})
    return {"ok": True}


@router.post("/models")
async def save_model(request: Request, state: AppState = Depends(get_state)):
    payload = await parsed_body_or_error(request)
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        return JSONResponse({"error": "model_id is required"}, status_code=400)
    pricing = payload.get("pricing") or ""
    if pricing and isinstance(pricing, str):
        try:
            json.loads(pricing)
        except json.JSONDecodeError as exc:
            return JSONResponse({"error": f"pricing must be JSON: {exc.msg}"}, status_code=400)
    state.db.upsert_model_meta(model_id, payload.get("description") or None, pricing or None)
    state.models = load_catalog(state.settings.models_config, state.db)
    state.hub.publish({"type": "models-changed"})
    return {"ok": True}


@router.post("/models/{model_id}/delete")
async def delete_model(model_id: str, state: AppState = Depends(get_state)):
    state.db.delete_model_meta(model_id)
    state.models = load_catalog(state.settings.models_config, state.db)
    state.hub.publish({"type": "models-changed"})
    return {"ok": True}
