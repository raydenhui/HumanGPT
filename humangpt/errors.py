"""Error types and OpenAI-shaped error envelopes.

Every OpenAI-compatible endpoint returns errors as::

    {"error": {"message": str, "type": str, "param": null|str, "code": null|str}}

with a matching HTTP status. SDKs (openai-python, etc.) raise these as
exceptions on the client side.
"""

from __future__ import annotations

from typing import Any


class HumanGPTError(Exception):
    """Base class for expected application errors."""


class OpenAIErrorHeld(HumanGPTError):
    """Raised by endpoints to produce an OpenAI-shaped error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class RequestTimeoutError(OpenAIErrorHeld):
    """A parked request sat in the human queue longer than configured."""

    def __init__(self, timeout_s: int) -> None:
        super().__init__(
            f"Request timed out after {timeout_s}s waiting in the human queue",
            status_code=504,
            error_type="server_error",
            code="request_timeout",
        )


class InterruptedError(OpenAIErrorHeld):
    """A parked request was released because the server is shutting down."""

    def __init__(self) -> None:
        super().__init__(
            "Request interrupted: the HumanGPT server is shutting down before "
            "the operator could answer",
            status_code=503,
            error_type="api_connection_error",
            code="server_interrupted",
        )


class RequestNotFoundError(OpenAIErrorHeld):
    def __init__(self, request_id: str) -> None:
        super().__init__(
            f"No such request: {request_id}",
            status_code=404,
            error_type="invalid_request_error",
            code="request_not_found",
        )


class RequestNotPendingError(OpenAIErrorHeld):
    def __init__(self, request_id: str, state: str) -> None:
        super().__init__(
            f"Request {request_id} is not pending (current state: {state}); "
            "only pending requests can be answered",
            status_code=409,
            error_type="invalid_request_error",
            code="request_not_pending",
        )


class DiscardedError(OpenAIErrorHeld):
    """A parked request was released because the operator discarded it."""

    def __init__(self) -> None:
        super().__init__(
            "Request discarded by the operator",
            status_code=503,
            error_type="api_connection_error",
            code="request_discarded",
        )
