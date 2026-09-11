"""Model catalog: merge the configured model list with user-editable metadata.

The base model *identity* comes from the models.json file (path in
``MODELS_CONFIG``); user-editable ``description``/``pricing`` (and wholly new
model ids) live in the SQLite ``models_meta`` table. ``settings`` stores the
user's stream defaults (stream-mode + chunk-delay) so the Settings panel can
persist them.

The catalog is loaded once into ``AppState.models`` (list of dicts) at startup,
but the Settings panel writes through to the DB at edit time; a boot re-merge
picks those up. /v1/models always serves the merged result.
"""

from __future__ import annotations

import json
from typing import Any

from .db import Database
from .models_config import load_models


def load_catalog(path: Any, db: Database) -> list[dict[str, Any]]:
    """Load the base model list and overlay any user-edited metadata."""
    models = load_models(path)
    meta = {m["model_id"]: m for m in db.list_models_meta()}
    by_id = {m["id"]: m for m in models}
    for meta_id, meta_row in meta.items():
        if meta_id not in by_id:
            by_id[meta_id] = {
                "id": meta_id,
                "object": "model",
                "created": 1715367049,
                "owned_by": "humangpt",
            }
            models.append(by_id[meta_id])
        entry = by_id[meta_id]
        if meta_row["description"]:
            entry["description"] = meta_row["description"]
        if meta_row["pricing"]:
            try:
                entry["pricing"] = json.loads(meta_row["pricing"])
            except (ValueError, TypeError):  # pragma: no cover - stored JSON validated on write
                entry["pricing"] = None
    return models


def load_ui_settings(db: Database, settings: Any) -> dict[str, Any]:
    """Resolve the settings-panel defaults, layered over ``Settings``."""
    stream_mode = db.get_setting("stream_mode", settings.stream_mode) or "word-chunk"
    chunk_delay = db.get_setting("stream_chunk_delay_ms", str(settings.stream_chunk_delay_ms))
    try:
        chunk_delay_ms = int(chunk_delay)
    except (ValueError, TypeError):
        chunk_delay_ms = settings.stream_chunk_delay_ms
    return {"stream_mode": stream_mode, "stream_chunk_delay_ms": chunk_delay_ms}


def apply_ui_settings(db: Database, ui: dict[str, Any]) -> dict[str, Any]:
    """Persist the settings-panel selection; returns the canonical values."""
    stream_mode = ui.get("stream_mode")
    if stream_mode not in {"word-chunk", "once"}:
        stream_mode = "word-chunk"
    chunk_delay = int(ui.get("stream_chunk_delay_ms", 0) or 0)
    if chunk_delay < 0:
        chunk_delay = 0
    db.set_setting("stream_mode", stream_mode)
    db.set_setting("stream_chunk_delay_ms", str(chunk_delay))
    return {"stream_mode": stream_mode, "stream_chunk_delay_ms": chunk_delay}
