"""FastAPI application factory.

Composition root: builds Settings -> Database -> QueueService(+ParkingLot) ->
AppState, mounts the /v1 API and the web UI, and wires lifecycle (startup
orphan sweep, timeout sweep, graceful shutdown that interrupts parked
requests).
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import routers
from .api import routes as api_routes
from .catalog import load_catalog
from .config import Settings, load_settings
from .container import AppState
from .db import Database
from .errors import OpenAIErrorHeld
from .hub import BroadcastHub
from .models_config import load_models  # noqa: F401  (re-exported for embedders)
from .parking import InProcessParkingLot, QueueService
from .web import routes as web_routes

_WEB_DIR = Path(__file__).parent / "web"


def create_app(
    settings: Settings | None = None,
    *,
    dotenv_path: Path | None = None,
    install_signal_hook: bool = False,
) -> FastAPI:
    """Build the app.

    ``settings``/``dotenv_path`` are for embedding in tests.
    ``install_signal_hook`` — set by the ``humangpt`` entry point — installs
    a SIGTERM/SIGINT handler that interrupts parked requests *before* uvicorn
    waits for open connections to close. Without it, uvicorn's graceful
    shutdown would block forever on parked (long-lived) client connections,
    because its connection-wait happens before lifespan shutdown.
    """
    settings = settings or load_settings(dotenv_path=dotenv_path)

    db = Database(settings.db_path)

    # Startup sweep: rows left "pending" by a previous process are orphaned
    # (the old process is gone; nobody can answer them in-memory). Mark them
    # interrupted so the state machine stays honest.
    db.mark_stale_pending_interrupted(time.time())

    queue = QueueService(db, InProcessParkingLot())
    hub = BroadcastHub()

    models = load_catalog(settings.models_config, db)

    state = AppState(
        settings=settings,
        db=db,
        queue=queue,
        models=models,
        hub=hub,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue.on_release = lambda req_id, st: hub.publish(
            {"type": "request-changed", "id": req_id, "state": st}
        )
        await queue.start_timeout_sweep()

        signal_hooks = (
            _install_signal_hooks(state) if install_signal_hook else None
        )
        try:
            yield
        finally:
            # Graceful shutdown: release any still-parked request with an
            # interrupted error (503) instead of a hung socket, then run a
            # short bounded drain so in-flight handler tasks that are between
            # "task start" and "row created" finish before the DB closes.
            if signal_hooks is not None:
                signal_hooks.restore()
            await _interrupt_all_parked(state)
            await queue.stop_timeout_sweep()
            at = time.time()
            for _ in range(20):
                await asyncio.sleep(0.01)
                pending = db.list_requests(states=["pending"], limit=100000)
                changed = False
                for req in pending:
                    db.update_request_state(req.id, "interrupted", interrupted_at=at)
                    queue.resolve(req.id, answer=None, reason="interrupted")
                    changed = True
                if not changed:
                    await asyncio.sleep(0.01)
                    if not db.list_requests(states=["pending"], limit=100000):
                        break
            db.close()

    app = FastAPI(
        title="HumanGPT",
        version="0.1.0",
        description="Human-in-the-loop OpenAI-compatible mock endpoint. "
        "No AI generates responses — a human answers every request.",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _install_api_key_middleware(app, settings)

    @app.exception_handler(OpenAIErrorHeld)
    async def _openai_errors(request: Request, exc: OpenAIErrorHeld) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.state.app_state = state

    app.include_router(routers.v1_router, prefix="/v1")
    app.include_router(api_routes.router)
    app.include_router(web_routes.router)
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    return app


async def _interrupt_all_parked(state: AppState) -> None:
    """Transition every pending request to 'interrupted' and release its waiter.

    Idempotent — safe to call from both the shutdown signal hook and lifespan
    shutdown. The 503 responses let uvicorn's connection-wait drain quickly.
    """
    at = time.time()
    for req in state.db.list_requests(states=["pending"], limit=100000):
        state.db.update_request_state(req.id, "interrupted", interrupted_at=at)
        state.queue.resolve(req.id, answer=None, reason="interrupted")


class _SignalHooks:
    """SIGTERM/SIGINT hook that interrupts parked requests before uvicorn's
    connection-wait deadlock (see ``_install_signal_hooks``)."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._saved: dict[int, Any] = {}

    def install(self) -> None:
        import signal

        for signum in (signal.SIGTERM, signal.SIGINT):
            self._saved[signum] = signal.getsignal(signum)
        loop = asyncio.get_event_loop()
        state = self._state

        def handler(signum: int, frame) -> None:
            # Signal handlers run in the main thread between loop iterations;
            # schedule the interrupt work onto the loop, then defer to
            # uvicorn's own handler so it proceeds with its graceful sequence.
            try:
                loop.call_soon(
                    lambda: asyncio.create_task(_interrupt_all_parked(state))
                )
            except RuntimeError:  # pragma: no cover - loop already closing
                pass
            old = self._saved.get(signum)
            if callable(old):
                old(signum, frame)

        for signum in self._saved:
            signal.signal(signum, handler)

    def restore(self) -> None:
        import signal

        for signum, old in self._saved.items():
            signal.signal(signum, old)


def _install_signal_hooks(state: AppState) -> _SignalHooks:
    """Install the shutdown hook and return a handle to restore handlers later."""

    hooks = _SignalHooks(state)
    hooks.install()
    return hooks


def _install_api_key_middleware(app: FastAPI, settings: Settings) -> None:
    """Optional static API-key check for /v1 routes (never for the UI)."""

    fixed_key = settings.openai_api_key

    if not fixed_key:
        return

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[len("Bearer "):].strip() != fixed_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Incorrect API key provided",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def assert_single_worker() -> None:
    """Fail fast when the process looks multi-worker.

    The parked-request mechanism is an in-process asyncio table; multiple
    uvicorn workers each hold their own table and would strand clients across
    workers. ``__main__`` forces workers=1; this is a belt-and-braces check.
    """
    from_web_concurrency = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if from_web_concurrency > 1:
        raise SystemExit(
            "HumanGPT requires a single uvicorn worker (parked requests live "
            "in-process). Found WEB_CONCURRENCY > 1."
        )
