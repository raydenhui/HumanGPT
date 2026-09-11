"""In-process publish/subscribe hub for the operator UI.

The web UI keeps an SSE connection open to /events. Rather than polling SQLite,
the action handlers publish a small change event to this hub and all connected
browsers receive it within ~1s. Single-worker only (same process as the queue),
which matches the deployment model. Backed by asyncio.Queue per subscriber so
slow clients don't block publishers.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any


class BroadcastHub:
    def __init__(self, max_queue: int = 100) -> None:
        self._listeners: dict[int, asyncio.Queue[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._max_queue = max_queue

    def subscribe(self) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        listener_id = next(self._ids)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        self._listeners[listener_id] = queue
        return listener_id, queue

    def unsubscribe(self, listener_id: int) -> None:
        self._listeners.pop(listener_id, None)

    def publish(self, event: dict[str, Any]) -> None:
        """Fire-and-forget: drop the event for slow/overflowing listeners."""
        for queue in list(self._listeners.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover
                pass
