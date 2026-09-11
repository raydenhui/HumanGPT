"""Unit tests for humangpt/errors.py and humangpt/endpoint_utils.build_request_row."""

from __future__ import annotations

import json

from humangpt.endpoint_utils import build_request_row, new_request_id
from humangpt.errors import (
    DiscardedError,
    HumanGPTError,
    InterruptedError,
    OpenAIErrorHeld,
    RequestNotFoundError,
    RequestNotPendingError,
    RequestTimeoutError,
)


class TestErrorPayloads:
    def test_openai_error_shape(self):
        err = OpenAIErrorHeld("bad", status_code=422, error_type="invalid_request_error", param="messages", code="x")
        payload = err.to_payload()
        assert payload == {
            "error": {"message": "bad", "type": "invalid_request_error", "param": "messages", "code": "x"}
        }
        assert err.status_code == 422

    def test_defaults(self):
        err = OpenAIErrorHeld("oops")
        assert err.status_code == 400
        assert err.error_type == "invalid_request_error"
        assert err.param is None
        assert err.code is None

    def test_timeout_error(self):
        err = RequestTimeoutError(30)
        assert err.status_code == 504
        assert err.error_type == "server_error"
        assert err.code == "request_timeout"
        assert "30s" in err.message

    def test_interrupted_error(self):
        err = InterruptedError()
        assert err.status_code == 503
        assert err.code == "server_interrupted"

    def test_discarded_error(self):
        err = DiscardedError()
        assert err.status_code == 503
        assert err.code == "request_discarded"

    def test_not_found_error(self):
        err = RequestNotFoundError("req_1")
        assert err.status_code == 404
        assert "req_1" in err.message

    def test_not_pending_error(self):
        err = RequestNotPendingError("req_1", "answered")
        assert err.status_code == 409
        assert "answered" in err.message

    def test_all_are_humangpt_errors(self):
        for err in (RequestTimeoutError(1), InterruptedError(), DiscardedError(), RequestNotFoundError("x")):
            assert isinstance(err, HumanGPTError)
            assert isinstance(err, OpenAIErrorHeld)


class TestBuildRequestRow:
    def test_builds_pending_row_with_timeout(self):
        row = build_request_row(
            endpoint="/v1/chat/completions",
            model="human-gpt",
            body_raw={"model": "human-gpt", "messages": []},
            parsed_summary={"prompt_tokens": 3},
            stream_mode="word-chunk",
            timeout_s=60,
            request_id="req_test",
            created_at=1000.0,
        )
        assert row.id == "req_test"
        assert row.endpoint == "/v1/chat/completions"
        assert row.model == "human-gpt"
        assert row.state == "pending"
        assert row.stream_mode == "word-chunk"
        assert row.created_at == 1000.0
        assert row.timeout_at == 1060.0
        assert json.loads(row.body)["model"] == "human-gpt"
        assert row.parsed_json["prompt_tokens"] == 3

    def test_timeout_none_when_disabled(self):
        row = build_request_row(
            endpoint="/v1/responses",
            model="human-gpt",
            body_raw={"input": "hi"},
            parsed_summary={},
            stream_mode="once",
            timeout_s=None,
        )
        assert row.timeout_at is None

    def test_timeout_none_when_zero(self):
        row = build_request_row(
            endpoint="/v1/chat/completions",
            model="m",
            body_raw={},
            parsed_summary={},
            stream_mode="word-chunk",
            timeout_s=0,
        )
        assert row.timeout_at is None

    def test_new_request_id_shape(self):
        rid = new_request_id()
        assert rid.startswith("req_")
        assert len(rid) == len("req_") + 24
        assert new_request_id() != rid
