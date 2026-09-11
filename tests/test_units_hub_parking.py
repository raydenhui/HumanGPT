"""Unit tests for humangpt/hub.py and humangpt/parking.py."""

from __future__ import annotations

import asyncio

import pytest

from humangpt.db import Database, RequestRow
from humangpt.errors import RequestTimeoutError
from humangpt.hub import BroadcastHub
from humangpt.parking import InProcessParkingLot, ParkedResult, QueueService

# ---------------------------------------------------------------- hub


async def test_hub_subscribe_publish_unsubscribe():
    hub = BroadcastHub()
    listener_id, queue = hub.subscribe()
    hub.publish({"type": "x"})
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"type": "x"}
    hub.unsubscribe(listener_id)
    hub.publish({"type": "y"})
    assert queue.empty()


async def test_hub_fans_out_to_all_listeners():
    hub = BroadcastHub()
    _id1, q1 = hub.subscribe()
    _id2, q2 = hub.subscribe()
    hub.publish({"n": 1})
    assert (await asyncio.wait_for(q1.get(), 1)) == {"n": 1}
    assert (await asyncio.wait_for(q2.get(), 1)) == {"n": 1}


async def test_hub_drops_oldest_on_overflow():
    hub = BroadcastHub(max_queue=2)
    _id, queue = hub.subscribe()
    hub.publish({"n": 1})
    hub.publish({"n": 2})
    hub.publish({"n": 3})  # overflow -> oldest (n=1) dropped
    got = [await asyncio.wait_for(queue.get(), 1) for _ in range(2)]
    assert got == [{"n": 2}, {"n": 3}]


# ------------------------------------------------------------- parking


async def _collect(lot, request_id, timeout_s):
    out = []
    async for kind, value in lot.iter_results(request_id, timeout_s):
        out.append((kind, value))
    return out


async def test_iter_results_yields_deltas_then_result():
    lot = InProcessParkingLot()
    task = asyncio.create_task(_collect(lot, "req", 5))
    await asyncio.sleep(0)  # let the generator register its queue
    lot.emit("req", {"type": "delta", "delta": "Hello "})
    lot.emit("req", {"type": "delta", "delta": "world"})
    lot.resolve("req", {"response_id": "r1"}, "answered")
    out = await asyncio.wait_for(task, 2)
    kinds = [k for k, _ in out]
    assert kinds == ["delta", "delta", "result"]
    deltas = [v["delta"] for k, v in out if k == "delta"]
    assert "".join(deltas) == "Hello world"
    result = out[-1][1]
    assert isinstance(result, ParkedResult)
    assert result.answer == {"response_id": "r1"}
    assert result.reason == "answered"


async def test_iter_results_timeout_raises():
    lot = InProcessParkingLot()
    with pytest.raises(RequestTimeoutError):
        await _collect(lot, "req", 0.05)


async def test_emit_without_waiter_is_noop():
    lot = InProcessParkingLot()
    lot.emit("nobody", {"type": "delta", "delta": "x"})  # must not raise


async def test_resolve_without_waiter_is_noop():
    lot = InProcessParkingLot()
    lot.resolve("nobody", {"ok": True}, "answered")  # must not raise


async def test_resolve_marks_interrupt_reason():
    lot = InProcessParkingLot()
    task = asyncio.create_task(_collect(lot, "req", 5))
    await asyncio.sleep(0)
    lot.resolve("req", None, "interrupted")
    out = await asyncio.wait_for(task, 2)
    assert out[-1][1].answer is None
    assert out[-1][1].reason == "interrupted"


# -------------------------------------------------------- queue service


def make_request(rid="req_1", timeout_at=50.0) -> RequestRow:
    return RequestRow(
        id=rid,
        endpoint="/v1/chat/completions",
        model="human-gpt",
        body="{}",
        parsed='{"prompt_tokens":1}',
        state="pending",
        created_at=10.0,
        answered_at=None,
        answered_by=None,
        claimed_by=None,
        claimed_at=None,
        interrupted_at=None,
        stream_mode="word-chunk",
        timeout_at=timeout_at,
    )


def test_queue_service_park_consumes_result():
    db = Database(":memory:")
    db.create_request(make_request())
    service = QueueService(db, InProcessParkingLot())

    async def run():
        task = asyncio.create_task(service.park("req_1", 5))
        await asyncio.sleep(0)
        service.resolve("req_1", {"response_id": "r"}, "answered")
        return await asyncio.wait_for(task, 2)

    result = asyncio.run(run())
    assert result.answer == {"response_id": "r"}
    db.close()


def test_sweep_expired_releases_past_deadline():
    db = Database(":memory:")
    db.create_request(make_request("req_old", timeout_at=50.0))
    db.create_request(make_request("req_future", timeout_at=10_000.0))
    service = QueueService(db, InProcessParkingLot())
    released = []
    service.on_release = lambda rid, state: released.append((rid, state))

    result = service.sweep_expired(now=100.0)
    assert result == ["req_old"]
    assert db.get_request("req_old").state == "timed_out"
    assert db.get_request("req_future").state == "pending"
    assert released == [("req_old", "timed_out")]
    db.close()


def test_sweep_expired_skips_requests_without_deadline():
    db = Database(":memory:")
    db.create_request(make_request("req_none", timeout_at=None))
    service = QueueService(db, InProcessParkingLot())
    assert service.sweep_expired(now=10_000.0) == []
    assert db.get_request("req_none").state == "pending"
    db.close()
