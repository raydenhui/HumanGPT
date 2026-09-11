"""Token-count heuristics for realistic ``usage`` fields.

OpenAI's tokenization is opaque and model-specific, so we approximate. The
heuristics here are deliberately simple, deterministic, and documented — the
goal is *shape realism* for SDK consumers (some SDKs sum ``prompt_tokens`` +
``completion_tokens`` and some assert on ``total_tokens``), not exact parity
with a tokenizer.

Rules (documented in the README):
- prompt tokens:  ratio of encoded characters (``len(text.encode("utf-8"))``)
  to a fixed 4 bytes-per-token, bottom-capped at 1. Each message in the chain
  contributes its encoded length; tool/function definitions are counted on
  their JSON encoding (incl. whitespace, uncompressed).
- completion tokens: same 4-bytes-per-token rule on the human's answer text.
- total: prompt + completion.
"""

from __future__ import annotations

import json
from typing import Any

BYTES_PER_TOKEN = 4.0


def encode_to_tokens(text: str) -> int:
    """Chars/bytes -> estimated tokens (>=1)."""
    return max(1, int(len(text.encode("utf-8")) / BYTES_PER_TOKEN) + 1)


def _count_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += encode_to_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    total += encode_to_tokens(part["text"])
                elif part.get("type") in {"image_url", "input_image"}:
                    url = ""
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    elif isinstance(image_url, str):
                        url = image_url
                    if url.startswith("data:"):
                        total += 765  # heuristic: a base64 data-URI image ~= 765 tokens
                    else:
                        total += 85  # heuristic: a referenced http(s) image ~= 85 tokens
        elif msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                total += encode_to_tokens(json.dumps(fn))
        if isinstance(msg.get("name"), str):
            total += encode_to_tokens(msg["name"])
    return total


def _count_input_items(items: list[dict[str, Any]]) -> int:
    total = 0
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            total += encode_to_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    total += encode_to_tokens(part["text"])
                elif isinstance(part, dict) and part.get("type") == "input_image":
                    url = part.get("image_url", "")
                    total += 765 if str(url).startswith("data:") else 85
        if isinstance(item.get("output"), str):
            total += encode_to_tokens(item["output"])
    return total


def _count_tools(tools: list[dict[str, Any]] | None) -> int:
    if not tools:
        return 0
    return sum(encode_to_tokens(json.dumps(tool, separators=(",", ":"))) for tool in tools)


def estimate_chat_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    instructions: str | None = None,
) -> int:
    total = _count_messages(messages)
    total += _count_tools(tools)
    if instructions:
        total += encode_to_tokens(instructions)
    return max(1, total)


def estimate_responses_prompt_tokens(
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    instructions: str | None = None,
) -> int:
    total = _count_input_items(input_items)
    total += _count_tools(tools)
    if instructions:
        total += encode_to_tokens(instructions)
    return max(1, total)


def estimate_completion_tokens(assistant_text: str) -> int:
    return encode_to_tokens(assistant_text)
