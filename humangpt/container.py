"""Application container: wires config, db, queue and model list together.

Held on ``app.state.app_state``; endpoints and the web UI receive it through
FastAPI dependency injection (see ``humangpt/app.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .db import Database
from .hub import BroadcastHub
from .parking import QueueService


@dataclass(slots=True)
class AppState:
    settings: Settings
    db: Database
    queue: QueueService
    models: list[dict]
    hub: BroadcastHub
