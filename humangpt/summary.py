"""Parsed-summary builder: the compact, operator-friendly view of a request.

Stored in ``requests.parsed`` (JSON) at park time and rendered by the web UI.
Kept separate from the raw ``body`` so the raw bytes are always available for
the "raw JSON" toggle and for auditing.
"""

from __future__ import annotations

import json
from typing import Any

from . import tokens


def summarize_chat(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    preview = _first_text_preview(messages)
    return {
        "kind": "chat",
        "endpoint": "/v1/chat/completions",
        "prompt_tokens": tokens.estimate_chat_prompt_tokens(
            messages,
            body.get("tools"),
            None,
        ),
        "role_counts": _role_counts(messages),
        "has_images": _has_images(messages),
        "tool_names": _tool_names(body.get("tools")),
        "first_user_text": preview,
        "snippet": preview[:120],
        "params": _params_subset(body),
        "message_count": len(messages),
    }


def summarize_responses(body: dict[str, Any]) -> dict[str, Any]:
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        items: list[dict[str, Any]] = [{"type": "message", "role": "user", "content": raw_input}]
        input_items = json.dumps(raw_input)
        preview = raw_input[:120]
        has_images = False
        role_counts: dict[str, int] = {"user": 1}
        message_count = 1
    else:
        items = [item for item in raw_input or [] if isinstance(item, dict)]
        input_items = json.dumps(items)
        preview = _responses_preview(items)
        has_images = any(
            isinstance(p, dict) and p.get("type") == "input_image"
            for item in items
            for p in (item.get("content") or [])
            if isinstance(p, dict)
        )
        role_counts = {
            "user": sum(1 for i in items if i.get("role", "user") == "user"),
            "assistant": sum(1 for i in items if i.get("role") == "assistant"),
        }
        message_count = len(items)

    return {
        "kind": "responses",
        "endpoint": "/v1/responses",
        "prompt_tokens": tokens.estimate_responses_prompt_tokens(
            items,
            body.get("tools"),
            body.get("instructions"),
        ),
        "role_counts": role_counts,
        "has_images": has_images,
        "tool_names": _tool_names(body.get("tools")),
        "first_user_text": preview,
        "snippet": preview[:120],
        "params": _params_subset(body),
        "message_count": message_count,
        "input_preview": input_items[:500],
    }


def _params_subset(body: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model",
        "max_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "stream",
        "n",
        "response_format",
        "tool_choice",
        "top_logprobs",
        "logprobs",
        "user",
    ]
    return {k: body[k] for k in keys if k in body and body[k] is not None}


def _role_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in messages:
        role = m.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            fn = tool.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
    return names


def _first_text_preview(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") not in {"user", "system"}:
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip():
                    return str(part["text"])
    return ""


def _responses_preview(items: list[dict[str, Any]]) -> str:
    for item in items:
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"} and str(
                    part.get("text", "")
                ).strip():
                    return str(part["text"])
    return ""


def _has_images(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
                if isinstance(part, dict) and part.get("type") == "input_image":
                    return True
    return False
