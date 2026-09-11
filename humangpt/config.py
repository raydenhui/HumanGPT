"""Environment-driven configuration.

Uses a tiny hand-rolled loader (no pydantic-settings dependency): read a
``.env`` file if present, then let process environment variables override it,
then apply defaults. All names are ``HUMANGPT``-free and match the OpenAI-ecosystem
convention (``OPENAI_API_KEY``), so the file documents itself against
``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS: dict[str, str] = {
    "HOST": "127.0.0.1",
    "PORT": "8000",
    "DB_PATH": "data/humangpt.db",
    "OPENAI_API_KEY": "",
    "MOCK_REQUEST_TIMEOUT_S": "600",
    "MOCK_STREAM_CHUNK_DELAY_MS": "0",
    "MOCK_STREAM_MODE": "word-chunk",
    "MODELS_CONFIG": "models.json",
    "CORS_ORIGINS": "*",
    "UI_SSE_KEEPALIVE_S": "25",
    "UI_POLL_INTERVAL_S": "10",
}

_INT_KEYS = {
    "MOCK_REQUEST_TIMEOUT_S",
    "MOCK_STREAM_CHUNK_DELAY_MS",
    "UI_SSE_KEEPALIVE_S",
    "UI_POLL_INTERVAL_S",
}
_INT_NON_NEGATIVE = {"MOCK_REQUEST_TIMEOUT_S", "MOCK_STREAM_CHUNK_DELAY_MS"}


def _load_dotenv(path: Path | None) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` .env file (no interpolation/quoting).

    Returns a dict of raw values. Missing files are fine; unreadable files
    raise a clear error rather than silently misconfiguring the server.
    """
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"{path}:{lineno}: malformed line (expected KEY=VALUE): {raw!r}")
        out[key.strip()] = value.strip()
    return out


def _as_int(key: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - env misconfiguration
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if key in _INT_NON_NEGATIVE and value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _as_csv(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application configuration."""

    host: str = DEFAULTS["HOST"]
    port: int = int(DEFAULTS["PORT"])
    db_path: Path = Path(DEFAULTS["DB_PATH"])
    openai_api_key: str = ""
    request_timeout_s: int = int(DEFAULTS["MOCK_REQUEST_TIMEOUT_S"])
    stream_chunk_delay_ms: int = int(DEFAULTS["MOCK_STREAM_CHUNK_DELAY_MS"])
    stream_mode: str = DEFAULTS["MOCK_STREAM_MODE"]
    models_config: Path = Path(DEFAULTS["MODELS_CONFIG"])
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    ui_sse_keepalive_s: int = int(DEFAULTS["UI_SSE_KEEPALIVE_S"])
    ui_poll_interval_s: int = int(DEFAULTS["UI_POLL_INTERVAL_S"])

    @property
    def api_key_mode(self) -> str:
        """'fixed' when a key is configured, 'any' otherwise."""
        return "fixed" if self.openai_api_key else "any"

    def validate(self) -> Settings:
        if self.stream_mode not in {"word-chunk", "once"}:
            raise ValueError(
                f"MOCK_STREAM_MODE must be 'word-chunk' or 'once', got {self.stream_mode!r}"
            )
        if self.request_timeout_s < 0:
            raise ValueError("MOCK_REQUEST_TIMEOUT_S must be >= 0 (0 disables timeout)")
        if self.stream_chunk_delay_ms < 0:
            raise ValueError("MOCK_STREAM_CHUNK_DELAY_MS must be >= 0")
        return self


def load_settings(env: dict[str, str] | None = None, dotenv_path: Path | None = None) -> Settings:
    """Build Settings from (in priority order) explicit env, .env file, process env.

    Callers that pass ``dotenv_path`` explicitly also keep .env support even
    when running under pytest where the working directory may differ.
    """
    raw: dict[str, str] = {}

    file_values = _load_dotenv(dotenv_path)
    raw.update(file_values)

    effective_env = dict(os.environ if env is None else env)
    for key, value in effective_env.items():
        if key in DEFAULTS:
            raw[key] = value

    port = _as_int("PORT", raw.get("PORT", DEFAULTS["PORT"]))
    timeout = _as_int("MOCK_REQUEST_TIMEOUT_S", raw.get("MOCK_REQUEST_TIMEOUT_S", DEFAULTS["MOCK_REQUEST_TIMEOUT_S"]))
    delay = _as_int(
        "MOCK_STREAM_CHUNK_DELAY_MS",
        raw.get("MOCK_STREAM_CHUNK_DELAY_MS", DEFAULTS["MOCK_STREAM_CHUNK_DELAY_MS"]),
    )
    keepalive = _as_int("UI_SSE_KEEPALIVE_S", raw.get("UI_SSE_KEEPALIVE_S", DEFAULTS["UI_SSE_KEEPALIVE_S"]))
    poll = _as_int("UI_POLL_INTERVAL_S", raw.get("UI_POLL_INTERVAL_S", DEFAULTS["UI_POLL_INTERVAL_S"]))

    settings = Settings(
        host=raw.get("HOST", DEFAULTS["HOST"]),
        port=port,
        db_path=Path(raw.get("DB_PATH", DEFAULTS["DB_PATH"])),
        openai_api_key=raw.get("OPENAI_API_KEY", ""),
        request_timeout_s=timeout,
        stream_chunk_delay_ms=delay,
        stream_mode=raw.get("MOCK_STREAM_MODE", DEFAULTS["MOCK_STREAM_MODE"]),
        models_config=Path(raw.get("MODELS_CONFIG", DEFAULTS["MODELS_CONFIG"])),
        cors_origins=_as_csv(raw.get("CORS_ORIGINS", DEFAULTS["CORS_ORIGINS"])),
        ui_sse_keepalive_s=keepalive,
        ui_poll_interval_s=poll,
    )
    return settings.validate()
