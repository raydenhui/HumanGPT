"""Pure, unit-testable core of the HumanGPT operator flow.

This module holds the *logic* without I/O: state transitions, SSE/buffer
assembly, and answer-envelope construction. The API/web layers call into it;
the pure functions are directly testable (fast, no DB, no event loop).

It deliberately avoids importing fastapi/starlette/uvicorn so the unit tests
can import it with zero network dependencies.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

PENDING = "pending"
ANSWERED = "answered"
STREAMED = "streamed"
TIMED_OUT = "timed_out"
INTERRUPTED = "interrupted"
DISCARDED = "discarded"

ALL_STATES = (PENDING, ANSWERED, STREAMED, TIMED_OUT, INTERRUPTED, DISCARDED)

# Allowed transitions: {current: (target, ...)}
TRANSITIONS: dict[str, tuple[str, ...]] = {
    PENDING: (ANSWERED, STREAMED, TIMED_OUT, INTERRUPTED, DISCARDED),
    ANSWERED: (PENDING,),
    STREAMED: (PENDING,),
    TIMED_OUT: (PENDING,),
    INTERRUPTED: (PENDING,),
    DISCARDED: (PENDING,),
}


class InvalidTransition(Exception):
    def __init__(self, request_id: str, current: str, target: str) -> None:
        super().__init__(f"invalid transition {current} -> {target} for {request_id}")
        self.request_id = request_id
        self.current = current
        self.target = target


def can_transition(current: str, target: str) -> bool:
    """Whether the state machine allows ``current -> target``."""
    return target in TRANSITIONS.get(current, ())


def transition(current: str, target: str, request_id: str | None = None) -> str:
    """Validate a transition and return the target state, raising on illegal ones."""
    if not can_transition(current, target):
        raise InvalidTransition(request_id or "?", current, target)
    return target


def terminal_state(state: str) -> bool:
    return state in (ANSWERED, STREAMED, TIMED_OUT, INTERRUPTED, DISCARDED)


# ---------------------------------------------------------------------------
# Stream mode + word chunking (pure)
# ---------------------------------------------------------------------------

WORD_CHUNK = "word-chunk"
ONCE = "once"
STREAM_MODES = (WORD_CHUNK, ONCE)


def sanitize_stream_mode(mode: str | None, default: str = WORD_CHUNK) -> str:
    """Clamp to a known stream mode."""
    return mode if mode in STREAM_MODES else (default if default in STREAM_MODES else WORD_CHUNK)


def split_into_deltas(text: str, mode: str = WORD_CHUNK) -> list[str]:
    """Split a complete answer into the deltas a stream should emit.

    word-chunk: split on whitespace *keeping the separators* so
    ``"".join(deltas) == text`` exactly.
    once:       the whole text is one delta.
    """
    if mode == ONCE:
        return [text]
    if not text:
        return [text]
    parts = re.split(r"(\s+)", text)
    if not parts:
        return [text]
    # First pass: split the leading word from the first separator, then each
    # separator+word pair. Preserve exact reconstruction.
    out: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if chunk == "":
            i += 1
            continue
        if re.fullmatch(r"\s+", chunk):
            # A separator follows a word; attach it to the PREVIOUS delta so
            # the reconstruction is exact (word + its trailing whitespace).
            if out:
                out[-1] += chunk
            # else leading separator: attach to the next word? Keep it as its
            # own delta so join() is still faithful.
            else:
                out.append(chunk)
        else:
            out.append(chunk)
        i += 1
    # A leading-separator-only edge ("  hi") would make the first delta
    # whitespace; acceptable (join still = text).
    return out if "".join(out) == text or len(out) else [text]


def lock_word(draft: str) -> tuple[str, str]:
    """Live typing: given the draft text, return (locked_delta, remaining_tail).

    Mirrors the React SPA's word-lock semantics (``ui/src/wordlock.js``). The
    SPA's edit buffer at lock time is always ``[separators][word][trailing
    separators]`` (the tail holds only accumulated separators + the word being
    typed), so the whole-value regex applies. The locked delta is
    ``leading-separators + word`` (each separator consumed exactly once, riding
    as the next word's leading whitespace); the trailing separators stay in the
    tail so finish() can drop them — "1 2 3" → deltas ["1"," 2"," 3"], no
    double-spaces, no trailing space.

    Returns ``("", draft)`` when nothing is lockable yet (mid-word: no
    trailing separator, or an unfinished sequence).
    """
    m = re.fullmatch(r"(\s*)(\S+)(\s+)", draft)
    if not m:
        return "", draft
    lead, word, trail = m.group(1), m.group(2), m.group(3)
    return lead + word, trail


# ---------------------------------------------------------------------------
# Token heuristics (pure)
# ---------------------------------------------------------------------------

BYTES_PER_TOKEN = 4.0


def estimate_completion_tokens(text: str) -> int:
    return max(1, int(len(text.encode("utf-8")) / BYTES_PER_TOKEN) + 1)


# ---------------------------------------------------------------------------
# Answer envelope builders (pure; no DB/IO)
# ---------------------------------------------------------------------------


def build_text_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def build_tool_message(name: str, arguments: str, tool_call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def build_chat_usage(prompt_tokens: int, completion_text: str) -> dict[str, int]:
    comp = estimate_completion_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp,
        "total_tokens": prompt_tokens + comp,
    }


def build_response_envelope(
    *,
    request_id: str,
    model: str,
    prompt_tokens: int,
    text: str = "",
    is_tool_call: bool = False,
    tool_name: str = "",
    tool_arguments: str = "",
) -> dict[str, Any]:
    """Construct the exact chat.completion envelope a parked client receives.

    Pure: no DB, no clock injection. Callers pass ``request_id``/``model``/
    ``prompt_tokens`` explicitly so it's trivially unit-testable.
    """
    usage = build_chat_usage(prompt_tokens, text)
    message = (
        build_tool_message(tool_name, tool_arguments, f"call_{request_id[:8]}")
        if is_tool_call
        else build_text_message(text)
    )
    return {
        "id": f"chatcmpl-human-{request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "humangpt_fp_000000",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if is_tool_call else "stop",
                "logprobs": None,
            }
        ],
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Interim-event buffer assembly (pure)
# ---------------------------------------------------------------------------

# The live stream consumes a FIFO of interim events. The *pure* assembly of a
# final "response text" from a sequence of delta events is here (unit-tested in
# isolation); the actual queue lives in humangpt/parking.py.


def assemble_text_from_deltas(deltas: list[str]) -> str:
    """The client-side reconstruction: join the delta texts."""
    return "".join(deltas)


@dataclass
class LiveBuffer:
    """In-memory accumulator mirroring what a live word-chunk stream delivers.

    Lets the operator UI and the endpoint agree on the exact text without
    depending on the real asyncio queue (used by tests and by the SPA's
    optimistic display).
    """

    deltas: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return assemble_text_from_deltas(self.deltas)

    def append_word(self, delta: str) -> str:
        self.deltas.append(delta)
        return self.text

    def finish_with_tail(self, tail: str) -> str:
        """Append the operator's not-yet-streamed tail; returns the full text."""
        if tail:
            self.deltas.append(tail)
        return self.text
