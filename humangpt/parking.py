"""Parking lot: how parked client HTTP requests wait for a human answer.

In the current single-process deployment each parked request is held as an
asyncio coroutine waiting on a per-request ``asyncio.Queue`` keyed by request
ID. The web/operator path resolves the request when an answer (or a timeout)
is recorded, waking the parked coroutine exactly once.

The ``ParkingLot`` protocol is the seam for a future multi-worker deployment:
swap in a DB+pub/sub-backed implementation behind the same interface. The
answer-producing side (web UI) and the answer-consuming side (endpoints) only
talk to this object, so a later change of deployment model is isolated to this
module and the composition root (``humangpt/app.py``).
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from .errors import RequestTimeoutError


class ParkedResult:
    """What a parked request receives when it wakes.

    ``answer`` is the operator-authored response dict (already persisted by the
    answer path), or ``None`` on timeout/interrupt (in which case the endpoint
    raises the appropriate error).
    """

    __slots__ = ("answer", "reason")

    def __init__(self, answer: dict[str, Any] | None, reason: str = "answered") -> None:
        self.answer = answer
        self.reason = reason


class ParkingLot(ABC):
    """Wait for / resolve parked client requests.

    A parked request may also receive an arbitrary number of *interim events*
    (``emit``) before the final answer — the live-typing stream uses this to
    push content deltas as the operator types. ``emit`` is fire-and-forget.
    """

    @abstractmethod
    def iter_results(
        self, request_id: str, timeout_s: int | None
    ) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
        """Yield ``("delta", {…})`` for each interim event, then ``("result",
        <ParkedResult>)`` as the terminal item.

        The streaming endpoint iterates this to forward deltas live; the
        non-streaming path consumes only the terminal item. Type of the
        terminal value is ParkedResult.
        """

    @abstractmethod
    def resolve(self, request_id: str, answer: dict[str, Any] | None, reason: str) -> None:
        """Wake the parked request. Safe to call more than once (no-op after first)."""

    @abstractmethod
    def emit(self, request_id: str, event: dict[str, Any]) -> None:
        """Queue an interim event to the parked request (live-typing deltas)."""


class _Waiter:
    """Async generator adapter: converts the abstract ``iter_results`` protocol
    into the concrete loop a parked endpoint runs.

    Not used directly — the concrete InProcessParkingLot implements the same
    method; this class only documents the shape.
    """


class InProcessParkingLot(ParkingLot):
    """asyncio-based parking: one FIFO ``asyncio.Queue`` per parked request.

    ``emit()`` appends a ``dict`` (delta); ``resolve()`` appends a
    ``ParkedResult``. Because both share one FIFO, any deltas emitted before
    resolve are always ordered ahead of the result — no polling, no lost
    events. ``iter_results`` drains the queue and yields deltas (and the final
    ``ParkedResult``) in arrival order, with a total deadline for the wait.

    Single-worker only.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Any]] = {}

    async def iter_results(
        self, request_id: str, timeout_s: int | None
    ) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
        if request_id in self._queues:
            raise RuntimeError(f"request {request_id} already parked")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._queues[request_id] = queue
        deadline = None if timeout_s is None or timeout_s <= 0 else time.monotonic() + timeout_s
        try:
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestTimeoutError(timeout_s or 0)
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except TimeoutError:
                    raise RequestTimeoutError(timeout_s or 0) from None
                if isinstance(item, ParkedResult):
                    yield "result", item
                    return
                yield "delta", item
        finally:
            self._queues.pop(request_id, None)

    def resolve(self, request_id: str, answer: dict[str, Any] | None, reason: str) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            return
        try:
            queue.put_nowait(ParkedResult(answer, reason))
        except asyncio.QueueFull:  # pragma: no cover - defensive
            pass

    def emit(self, request_id: str, event: dict[str, Any]) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - defensive
            pass


class QueueService:
    """Composition root for the human-in-the-loop flow.

    Owns the Database, the ParkingLot, and the per-request timeout sweep. This
    is what the endpoints and the web UI talk to — it couples persistence and
    wake-up semantics so neither side needs to know about the other.
    """

    def __init__(self, db: Any, parking_lot: ParkingLot) -> None:
        self.db = db
        self.parking_lot = parking_lot
        self._timeout_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.on_release: Any = None  # callable(request_id, state) — set by app wiring

    def iter_results(
        self, request_id: str, timeout_s: int | None
    ) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
        return self.parking_lot.iter_results(request_id, timeout_s)

    async def park(self, request_id: str, timeout_s: int | None) -> ParkedResult:
        """Blocking wait that only consumes the terminal result.

        Used by non-streaming endpoints (no deltas are emitted for them).
        """
        async for kind, value in self.parking_lot.iter_results(request_id, timeout_s):
            if kind == "result":
                return value  # type: ignore[return-value]
        raise RuntimeError("park() returned without a result")  # pragma: no cover

    def resolve(
        self,
        request_id: str,
        answer: dict[str, Any] | None = None,
        reason: str = "answered",
    ) -> None:
        self.parking_lot.resolve(request_id, answer, reason)

    def emit(self, request_id: str, event: dict[str, Any]) -> None:
        """Push an interim event (live-typing delta) to a parked request."""
        self.parking_lot.emit(request_id, event)

    # ------------------------------------------------------------ timeouts

    def sweep_expired(self, now: float | None = None, timeout_s: int | None = None) -> list[str]:
        """Find pending requests past their deadline and release them with a timeout."""
        now = now if now is not None else time.time()
        released: list[str] = []
        for req in self.db.list_requests(states=["pending"], limit=10000):
            deadline = req.timeout_at
            if deadline is None or deadline > now:
                continue
            if self.db.update_request_state(req.id, "timed_out", answered_at=now):
                self.parking_lot.resolve(req.id, answer=None, reason="timeout")
                self._notify_release(req.id, "timed_out")
                released.append(req.id)
        return released

    async def start_timeout_sweep(self, interval_s: float = 1.0) -> None:
        if self._timeout_task is not None:
            return
        self._stop_event.clear()
        self._timeout_task = asyncio.create_task(self._sweep_loop(interval_s))

    async def _sweep_loop(self, interval_s: float) -> None:
        while not self._stop_event.is_set():
            self.sweep_expired()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_s)
            except TimeoutError:
                pass

    async def stop_timeout_sweep(self) -> None:
        if self._timeout_task is None:
            return
        self._stop_event.set()
        self._timeout_task.cancel()  # pragma: no cover - defensive
        try:
            await self._timeout_task
        except (asyncio.CancelledError, TimeoutError):  # pragma: no cover
            pass
        self._timeout_task = None

    def _notify_release(self, request_id: str, state: str) -> None:
        if self.on_release is not None:
            try:
                self.on_release(request_id, state)
            except Exception:  # pragma: no cover - UI notification must never break the flow
                pass
