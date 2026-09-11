"""Public request/response models (Pydantic v2) for the OpenAI-compatible API.

These are the external contract: what we accept on ``POST /v1/chat/completions``
and ``POST /v1/responses``, and the envelopes we return on non-streaming calls.
The parsers intentionally accept the *real* OpenAI shapes (including tokenizers'
idiosyncrasies like an empty ``messages`` array and stray unknown fields, which
OpenAI itself tolerates) while rejecting bodies that are structurally wrong.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

# --------------------------------------------------------------------------
# Shared primitives
# --------------------------------------------------------------------------

ChatRole = Literal["system", "user", "assistant", "tool", "developer"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
ToolCallType = Literal["function"]
ResponseItemType = Literal["message", "function_call", "function_call_output"]


def _allow_extra(**kwargs: Any) -> ConfigDict:
    return ConfigDict(extra="allow", **kwargs)


class ImageUrl(BaseModel):
    model_config = _allow_extra()
    url: str = Field(..., description="http(s) or data: URI")
    detail: str | None = None


class TextPart(BaseModel):
    model_config = _allow_extra()
    type: Literal["text"] = "text"
    text: str


class ImageUrlPart(BaseModel):
    model_config = _allow_extra()
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl | str


class InputImagePart(BaseModel):
    """Responses-API vision input item (OpenAI calls it ``input_image``)."""

    model_config = _allow_extra()
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: str | None = None


class ContentPart(BaseModel):
    """Union: a single part of a ``content`` array in chat messages."""

    model_config = _allow_extra()

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | str | None = None
    detail: str | None = None

    @property
    def kind(self) -> Literal["text", "image_url"]:
        return self.type

    @property
    def image_url_value(self) -> str | None:
        if self.type != "image_url" or self.image_url is None:
            return None
        if isinstance(self.image_url, ImageUrl):
            return self.image_url.url
        return self.image_url

    @property
    def image_detail(self) -> str | None:
        if self.type != "image_url":
            return None
        if isinstance(self.image_url, ImageUrl):
            return self.image_url.detail
        return self.detail


def parse_content_parts(value: Any) -> list[ContentPart]:
    """Normalize a message ``content`` (str or list of parts) into parts.

    ``None`` (tool messages, some assistant messages) becomes [].
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [ContentPart(type="text", text=value, image_url=None)]
    if isinstance(value, list):
        return TypeAdapter(list[ContentPart]).validate_python(value)
    raise ValueError("content must be a string or an array of content parts")


# --------------------------------------------------------------------------
# Tool definitions (shared by chat completions and responses API)
# --------------------------------------------------------------------------


class FunctionDef(BaseModel):
    model_config = _allow_extra()
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ToolDef(BaseModel):
    model_config = _allow_extra()
    type: Literal["function"]
    function: FunctionDef


# --------------------------------------------------------------------------
# Chat completions request
# --------------------------------------------------------------------------


def _validate_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        for msg in value:
            if not isinstance(msg, dict):
                raise ValueError("each message must be an object")
            role = msg.get("role")
            if role not in {"system", "user", "assistant", "tool", "developer"}:
                raise ValueError(f"unsupported message role: {role!r}")
            if role == "user" and "content" not in msg:
                raise ValueError("user message missing 'content'")
    return value


class ChatCompletionsRequest(BaseModel):
    """Accepted subset of POST /v1/chat/completions.

    ``extra='allow'`` + tolerant parsing mirrors real OpenAI behaviour: unknown
    fields are ignored, and the SDK doesn't have to strip its own extras.
    """

    model_config = _allow_extra()

    model: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(..., description="message list, each entry a dict")
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    n: int | None = None

    def parsed_messages(self) -> list[dict[str, Any]]:
        return _validate_messages(self.messages)

    @property
    def include_usage(self) -> bool:
        opts = self.stream_options or {}
        return bool(opts.get("include_usage"))


class ChatRetrieved(BaseModel):
    """Normalized view the operator sees in the UI."""

    model_config = _allow_extra()
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: list[ContentPart]
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# --------------------------------------------------------------------------
# Responses API request
# --------------------------------------------------------------------------


class ResponseMessageItem(BaseModel):
    """One ``message`` item inside the Responses ``input`` array."""

    model_config = _allow_extra()
    type: Literal["message", "function_call_output"] = "message"
    role: Literal["assistant", "user", "system", "developer", "tool"] = "user"
    content: Any = None  # str | list[TextPart | ImageUrlPart | InputImagePart]

    def content_parts(self) -> list[ContentPart]:
        if isinstance(self.content, str):
            return [ContentPart(type="text", text=self.content, image_url=None)]
        if isinstance(self.content, list):
            return TypeAdapter(list[ContentPart]).validate_python(self.content)
        return []

    def _validate(self) -> None:
        self.content_parts()


class ResponseItem(BaseModel):
    """Union item; extra='allow' so unknown item shapes are tolerated just
    like OpenAI does (with a recorded warning)."""

    model_config = _allow_extra()
    type: str
    role: str | None = None
    content: Any = None
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None
    output: str | None = None


class ResponsesRequest(BaseModel):
    model_config = _allow_extra()

    model: str = Field(..., min_length=1)
    input: str | list[Any] = Field(..., description="string or array of input items")
    instructions: str | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stream: bool | None = None
    metadata: dict[str, Any] | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    include: list[Any] | None = None

    def parsed_input_items(self) -> list[dict[str, Any]]:
        """Return the input as a list of item dicts (string -> single user message)."""
        if isinstance(self.input, str):
            return [{"type": "message", "role": "user", "content": self.input}]
        items: list[dict[str, Any]] = []
        for item in self.input:
            if isinstance(item, str):
                items.append({"type": "message", "role": "user", "content": item})
            elif isinstance(item, dict):
                items.append(dict(item))
            else:
                raise ValueError(f"invalid input item: {item!r}")
        return items


# --------------------------------------------------------------------------
# Envelopes (non-streaming responses)
# --------------------------------------------------------------------------


class Choice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: FinishReason
    logprobs: dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: list[Choice]
    usage: Usage


class ResponsesEnvelope(BaseModel):
    model_config = _allow_extra()
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed"] = "completed"
    model: str
    output: list[dict[str, Any]]
    parallel_tool_calls: bool = True
    tool_choice: str | dict[str, Any] | None = None
    usage: dict[str, Any]
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Factory helpers
# --------------------------------------------------------------------------


def new_chat_completion(
    *,
    model: str,
    message: dict[str, Any],
    finish_reason: FinishReason,
    usage: dict[str, int],
    system_fingerprint: str = "humangpt_fp_000000",
    created: int | None = None,
    request_id: str | None = None,
) -> ChatCompletion:
    suffix = (request_id or uuid.uuid4().hex)[:8]
    return ChatCompletion(
        id=f"chatcmpl-human-{suffix}",
        object="chat.completion",
        created=created or int(time.time()),
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        usage=Usage.model_validate(usage),
    )


def new_responses_envelope(
    *,
    model: str,
    outputs: list[dict[str, Any]],
    usage: dict[str, Any],
    request_id: str | None = None,
    created: int | None = None,
) -> ResponsesEnvelope:
    suffix = (request_id or uuid.uuid4().hex)[:8]
    return ResponsesEnvelope(
        id=f"resp-human-{suffix}",
        object="response",
        created_at=created or int(time.time()),
        status="completed",
        model=model,
        output=outputs,
        tool_choice=None,
        usage=usage,
    )


def validation_error_payload(exc: ValidationError) -> tuple[int, dict[str, Any]]:
    """Turn a Pydantic ValidationError into an OpenAI invalid_request_error payload."""
    details = "; ".join(
        ".".join(str(loc) for loc in e["loc"]) + ": " + str(e["msg"]) if e["loc"] else e["msg"]
        for e in exc.errors()
    )
    return 400, {
        "error": {
            "message": f"Invalid request: {details}",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request",
        }
    }


__all__ = [
    "ChatRole",
    "FinishReason",
    "ImageUrl",
    "TextPart",
    "ImageUrlPart",
    "InputImagePart",
    "ContentPart",
    "parse_content_parts",
    "FunctionDef",
    "ToolDef",
    "ChatCompletionsRequest",
    "ResponsesRequest",
    "Choice",
    "Usage",
    "ChatCompletion",
    "ResponsesEnvelope",
    "new_chat_completion",
    "new_responses_envelope",
    "validation_error_payload",
]
