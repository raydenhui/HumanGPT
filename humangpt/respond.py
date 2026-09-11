"""Response builder: wrap a human answer in the correct OpenAI envelope.

Called by the web UI at submit time (to persist the canonical response) and by
the endpoints at release time (to serve the parked request). The persisted
``responses.body`` is the single source of truth for what the client receives,
so re-building from the row and streaming from the row always agree.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from . import tokens
from .db import RequestRow
from .models import ChatCompletion, ResponsesEnvelope, new_chat_completion, new_responses_envelope


@dataclass(frozen=True, slots=True)
class AnswerPayload:
    """What the operator submitted in the web UI."""

    text: str = ""
    is_tool_call: bool = False
    tool_name: str | None = None
    tool_arguments: str | None = None
    stream_mode: str = "word-chunk"

    @property
    def arguments_text(self) -> str:
        """Normalize tool arguments to their JSON string form."""
        if self.tool_arguments is None:
            return "{}"
        if isinstance(self.tool_arguments, str):
            return self.tool_arguments
        try:
            import json

            return json.dumps(self.tool_arguments, separators=(",", ":"))
        except TypeError:  # pragma: no cover - caller passed a weird object
            return "{}"


def _sanitize_web_stream_mode(value: str | None) -> str:
    return value if value in {"word-chunk", "once"} else "word-chunk"


def build_chat_response(
    row: RequestRow,
    answer: AnswerPayload,
    *,
    created: int | None = None,
) -> tuple[ChatCompletion, dict[str, Any]]:
    """Build a ChatCompletion envelope and the canonical response-record dict."""
    prompts = _parsed_prompt_tokens(row)
    completion_tokens = tokens.estimate_completion_tokens(answer.text)

    usage = {
        "prompt_tokens": prompts,
        "completion_tokens": completion_tokens,
        "total_tokens": prompts + completion_tokens,
    }

    if answer.is_tool_call:
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": answer.tool_name or "unknown_tool", "arguments": answer.arguments_text},
                }
            ],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": answer.text}
        finish_reason = "stop"

    completion = new_chat_completion(
        model=row.model,
        message=message,
        finish_reason=finish_reason,
        usage=usage,
        created=created,
        request_id=row.id,
    )
    record = {
        "stream_mode": _sanitize_web_stream_mode(answer.stream_mode),
        "is_tool_call": answer.is_tool_call,
        "tool_name": answer.tool_name if answer.is_tool_call else None,
        "tool_arguments": answer.arguments_text if answer.is_tool_call else None,
    }
    return completion, record


def build_responses_envelope(
    row: RequestRow,
    answer: AnswerPayload,
    *,
    created: int | None = None,
) -> tuple[ResponsesEnvelope, dict[str, Any]]:
    """Build a Responses envelope and the canonical response-record dict."""
    prompts = _parsed_prompt_tokens(row)
    completion_tokens = tokens.estimate_completion_tokens(answer.text)

    if answer.is_tool_call:
        fc_id = f"fc_{uuid.uuid4().hex[:16]}"
        outputs = [
            {
                "type": "function_call",
                "id": fc_id,
                "call_id": f"call_{uuid.uuid4().hex[:16]}",
                "name": answer.tool_name or "unknown_tool",
                "arguments": answer.arguments_text,
                "status": "completed",
            }
        ]
    else:
        outputs = [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:16]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": answer.text}],
            }
        ]

    usage = {
        "input_tokens": prompts,
        "output_tokens": completion_tokens,
        "total_tokens": prompts + completion_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }

    envelope = new_responses_envelope(
        model=row.model, outputs=outputs, usage=usage, request_id=row.id, created=created
    )
    record = {
        "stream_mode": _sanitize_web_stream_mode(answer.stream_mode),
        "is_tool_call": answer.is_tool_call,
        "tool_name": answer.tool_name if answer.is_tool_call else None,
        "tool_arguments": answer.arguments_text if answer.is_tool_call else None,
    }
    return envelope, record


def _parsed_prompt_tokens(row: RequestRow) -> int:
    parsed = row.parsed_json
    return int(parsed.get("prompt_tokens", 1)) or 1


def envelope_text(envelope: dict[str, Any]) -> str:
    """Extract the assistant text from a serialized envelope (for chunking)."""
    if envelope.get("object") == "chat.completion":
        message = envelope["choices"][0]["message"]
        if message.get("tool_calls"):
            return message["tool_calls"][0]["function"]["arguments"]
        return message.get("content") or ""
    if envelope.get("object") == "response":
        for out in envelope.get("output", []):
            if out.get("type") == "message":
                parts = out.get("content", [])
                if parts and parts[0].get("type") == "output_text":
                    return parts[0].get("text", "")
        return ""
    return ""


def is_tool_call_envelope(envelope: dict[str, Any]) -> bool:
    if envelope.get("object") == "chat.completion":
        return bool(envelope.get("choices", [{}])[0].get("message", {}).get("tool_calls"))
    if envelope.get("object") == "response":
        return any(o.get("type") == "function_call" for o in envelope.get("output", []))
    return False


def tool_call_parts(envelope: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return (tool_call_id, name, arguments) if the envelope carries a tool call."""
    if envelope.get("object") == "chat.completion":
        calls = envelope.get("choices", [{}])[0].get("message", {}).get("tool_calls")
        if calls:
            call = calls[0]
            return call.get("id", ""), call["function"]["name"], call["function"]["arguments"]
    if envelope.get("object") == "response":
        for out in envelope.get("output", []):
            if out.get("type") == "function_call":
                return out.get("id", ""), out.get("name", ""), out.get("arguments", "")
    return None
