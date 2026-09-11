"""Serving of the configured model list.

Reads a JSON file (path from config) shaped like::

    {"models": [{"id": "...", "object": "model", "created": 123, "owned_by": "..."}]}

The list is resolved once at startup (fail-fast). ``GET /v1/models`` then
returns it wrapped in the OpenAI envelope ``{"object": "list", "data": [...]}``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import HumanGPTError

MODEL_LIST_SCHEMA = (
    "models.json must contain {\"models\": [{\"id\": str, \"created\": int, "
    "\"owned_by\": str}, ...]}"
)


def load_models(path: Path) -> list[dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HumanGPTError(
            f"MODELS_CONFIG file not found: {path} ({MODEL_LIST_SCHEMA})"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HumanGPTError(
            f"MODELS_CONFIG file is not valid JSON: {path} ({exc.msg})"
        ) from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise HumanGPTError(MODEL_LIST_SCHEMA)

    models: list[dict] = []
    for entry in raw["models"]:
        if not isinstance(entry, dict):
            raise HumanGPTError(MODEL_LIST_SCHEMA)
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise HumanGPTError(MODEL_LIST_SCHEMA)
        created = entry.get("created", 1715367049)
        owned_by = entry.get("owned_by", "humangpt")
        models.append(
            {
                "id": model_id,
                "object": "model",
                "created": int(created),
                "owned_by": str(owned_by),
            }
        )
    return models
