"""Unit tests for humangpt/db.py — the SQLite persistence layer."""

from __future__ import annotations

import json

from humangpt.db import Database, RequestRow, ResponseRow


def make_db() -> Database:
    return Database(":memory:")


def make_request(rid="req_1", state="pending", created_at=100.0, timeout_at=200.0) -> RequestRow:
    return RequestRow(
        id=rid,
        endpoint="/v1/chat/completions",
        model="human-gpt",
        body=json.dumps({"stream": False, "model": "human-gpt"}),
        parsed=json.dumps({"prompt_tokens": 5, "snippet": "hi"}),
        state=state,
        created_at=created_at,
        answered_at=None,
        answered_by=None,
        claimed_by=None,
        claimed_at=None,
        interrupted_at=None,
        stream_mode="word-chunk",
        timeout_at=timeout_at,
    )


def make_response(rid="resp_1", request_id="req_1") -> ResponseRow:
    return ResponseRow(
        id=rid,
        request_id=request_id,
        body=json.dumps({"object": "chat.completion"}),
        finish_reason="stop",
        is_tool_call=False,
        tool_name=None,
        tool_arguments=None,
        stream_mode="word-chunk",
        created_at=101.0,
    )


class TestRequests:
    def test_create_and_get(self):
        db = make_db()
        db.create_request(make_request())
        row = db.get_request("req_1")
        assert row is not None
        assert row.model == "human-gpt"
        assert row.state == "pending"
        assert row.body_json["stream"] is False
        assert row.parsed_json["prompt_tokens"] == 5
        db.close()

    def test_get_missing_returns_none(self):
        db = make_db()
        assert db.get_request("nope") is None
        db.close()

    def test_list_by_state_and_order(self):
        db = make_db()
        db.create_request(make_request("req_2", created_at=200.0))
        db.create_request(make_request("req_1", created_at=100.0))
        db.create_request(make_request("req_3", state="answered", created_at=300.0))
        pending = db.list_requests(states=["pending"])
        assert [r.id for r in pending] == ["req_1", "req_2"]
        db.close()

    def test_count_by_state(self):
        db = make_db()
        db.create_request(make_request("r1"))
        db.create_request(make_request("r2"))
        db.create_request(make_request("r3", state="answered"))
        counts = db.count_by_state()
        assert counts == {"pending": 2, "answered": 1}
        db.close()

    def test_update_state_plain(self):
        db = make_db()
        db.create_request(make_request())
        assert db.update_request_state("req_1", "answered", answered_by="bob") is True
        row = db.get_request("req_1")
        assert row.state == "answered"
        assert row.answered_by == "bob"
        db.close()

    def test_update_state_only_if_pending_race(self):
        db = make_db()
        db.create_request(make_request())
        assert db.update_request_state("req_1", "answered", only_if_pending=True) is True
        # second attempt loses the race
        assert db.update_request_state("req_1", "answered", only_if_pending=True) is False
        db.close()

    def test_claim_and_unclaim(self):
        db = make_db()
        db.create_request(make_request())
        assert db.claim_request("req_1", "alice") is True
        assert db.get_request("req_1").claimed_by == "alice"
        # claim only applies to pending
        db.update_request_state("req_1", "answered")
        assert db.claim_request("req_1", "bob") is False
        db.close()

    def test_unclaim_only_by_owner(self):
        db = make_db()
        db.create_request(make_request())
        db.claim_request("req_1", "alice")
        assert db.unclaim_request("req_1", "bob") is False
        assert db.unclaim_request("req_1", "alice") is True
        assert db.get_request("req_1").claimed_by is None
        db.close()

    def test_discard(self):
        db = make_db()
        db.create_request(make_request())
        assert db.discard_request("req_1") is True
        assert db.get_request("req_1").state == "discarded"
        # cannot discard twice (already terminal)
        assert db.discard_request("req_1") is False
        db.close()

    def test_mark_stale_pending_interrupted(self):
        db = make_db()
        db.create_request(make_request("old", created_at=100.0))
        db.create_request(make_request("new", created_at=500.0))
        changed = db.mark_stale_pending_interrupted(before=300.0)
        assert changed == 1
        assert db.get_request("old").state == "interrupted"
        assert db.get_request("new").state == "pending"
        db.close()


class TestResponses:
    def test_create_and_get(self):
        db = make_db()
        db.create_request(make_request())
        db.create_response(make_response())
        resp = db.get_response("resp_1")
        assert resp is not None
        assert resp.body_json["object"] == "chat.completion"
        assert resp.finish_reason == "stop"

    def test_get_for_request(self):
        db = make_db()
        db.create_request(make_request())
        db.create_response(make_response())
        found = db.get_response_for_request("req_1")
        assert found.id == "resp_1"
        assert db.get_response_for_request("other") is None
        db.close()


class TestTemplates:
    def test_upsert_list_delete(self):
        db = make_db()
        db.create_template("t1", "body one")
        db.create_template("t2", "body two")
        assert [t.name for t in db.list_templates()] == ["t1", "t2"]
        # upsert overwrites body
        db.create_template("t1", "updated")
        assert db.get_template("t1").body == "updated"
        assert db.delete_template("t1") is True
        assert db.delete_template("t1") is False
        assert [t.name for t in db.list_templates()] == ["t2"]
        db.close()


class TestModelsMeta:
    def test_upsert_list_delete(self):
        db = make_db()
        db.upsert_model_meta("human-gpt", "desc", '{"input":1}')
        metas = db.list_models_meta()
        assert metas[0]["model_id"] == "human-gpt"
        assert metas[0]["description"] == "desc"
        db.upsert_model_meta("human-gpt", "desc2", None)
        assert db.list_models_meta()[0]["description"] == "desc2"
        assert db.delete_model_meta("human-gpt") is True
        assert db.list_models_meta() == []
        db.close()


class TestSettings:
    def test_get_set_delete(self):
        db = make_db()
        assert db.get_setting("k") is None
        assert db.get_setting("k", "default") == "default"
        db.set_setting("k", "v")
        assert db.get_setting("k") == "v"
        db.set_setting("k", "v2")
        assert db.get_setting("k") == "v2"
        db.delete_setting("k")
        assert db.get_setting("k") is None
        db.close()
