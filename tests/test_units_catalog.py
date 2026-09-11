"""Unit tests for humangpt/catalog.py — model catalog merge + UI settings."""

from __future__ import annotations

import json

from humangpt.catalog import apply_ui_settings, load_catalog, load_ui_settings
from humangpt.config import Settings
from humangpt.db import Database


def make_db() -> Database:
    return Database(":memory:")


def write_models(tmp_path, ids):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": [{"id": i, "created": 1, "owned_by": "humangpt"} for i in ids]}))
    return path


def test_load_catalog_base_only(tmp_path):
    db = make_db()
    catalog = load_catalog(write_models(tmp_path, ["human-gpt"]), db)
    assert [m["id"] for m in catalog] == ["human-gpt"]
    assert "description" not in catalog[0]
    db.close()


def test_load_catalog_overlays_description_and_pricing(tmp_path):
    db = make_db()
    db.upsert_model_meta("human-gpt", "A human-operated model", '{"input": 1}')
    catalog = load_catalog(write_models(tmp_path, ["human-gpt"]), db)
    entry = catalog[0]
    assert entry["description"] == "A human-operated model"
    assert entry["pricing"] == {"input": 1}
    db.close()


def test_load_catalog_adds_new_model_id(tmp_path):
    db = make_db()
    db.upsert_model_meta("claude-human", "added via panel", None)
    catalog = load_catalog(write_models(tmp_path, ["human-gpt"]), db)
    ids = [m["id"] for m in catalog]
    assert "claude-human" in ids
    db.close()


def test_load_catalog_ignores_invalid_pricing_json(tmp_path):
    db = make_db()
    # stored pricing that isn't valid JSON should not crash the merge
    db.upsert_model_meta("human-gpt", "d", "{not json")
    catalog = load_catalog(write_models(tmp_path, ["human-gpt"]), db)
    assert catalog[0].get("pricing") is None
    db.close()


def test_ui_settings_default_from_settings(tmp_path):
    db = make_db()
    settings = Settings(stream_mode="once", stream_chunk_delay_ms=42)
    ui = load_ui_settings(db, settings)
    assert ui == {"stream_mode": "once", "stream_chunk_delay_ms": 42}
    db.close()


def test_ui_settings_persisted_override(tmp_path):
    db = make_db()
    settings = Settings(stream_mode="once", stream_chunk_delay_ms=42)
    apply_ui_settings(db, {"stream_mode": "word-chunk", "stream_chunk_delay_ms": 7})
    ui = load_ui_settings(db, settings)
    assert ui == {"stream_mode": "word-chunk", "stream_chunk_delay_ms": 7}
    db.close()


def test_apply_ui_settings_clamps_invalid(tmp_path):
    db = make_db()
    result = apply_ui_settings(db, {"stream_mode": "bogus", "stream_chunk_delay_ms": -5})
    assert result == {"stream_mode": "word-chunk", "stream_chunk_delay_ms": 0}
    db.close()
