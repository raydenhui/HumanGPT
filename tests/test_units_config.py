"""Unit tests for humangpt/config.py and humangpt/models_config.py."""

from __future__ import annotations

import json

import pytest

from humangpt.config import _as_csv, _load_dotenv, load_settings
from humangpt.errors import HumanGPTError
from humangpt.models_config import load_models


class TestLoadSettings:
    def test_defaults(self):
        s = load_settings(env={})
        assert s.host == "127.0.0.1"
        assert s.port == 8000
        assert s.request_timeout_s == 600
        assert s.stream_mode == "word-chunk"
        assert s.cors_origins == ["*"]
        assert s.api_key_mode == "any"

    def test_env_overrides(self):
        s = load_settings(
            env={
                "HOST": "0.0.0.0",
                "PORT": "9999",
                "OPENAI_API_KEY": "sekrit",
                "MOCK_STREAM_MODE": "once",
                "MOCK_STREAM_CHUNK_DELAY_MS": "50",
                "CORS_ORIGINS": "http://a, http://b",
            }
        )
        assert s.host == "0.0.0.0"
        assert s.port == 9999
        assert s.api_key_mode == "fixed"
        assert s.stream_mode == "once"
        assert s.stream_chunk_delay_ms == 50
        assert s.cors_origins == ["http://a", "http://b"]

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError):
            load_settings(env={"PORT": "not-a-number"})

    def test_invalid_stream_mode_raises(self):
        with pytest.raises(ValueError):
            load_settings(env={"MOCK_STREAM_MODE": "bogus"})

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError):
            load_settings(env={"MOCK_REQUEST_TIMEOUT_S": "-1"})

    def test_dotenv_file_loaded(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("PORT=8123\nOPENAI_API_KEY=fromfile\n")
        s = load_settings(env={}, dotenv_path=env_file)
        assert s.port == 8123
        assert s.openai_api_key == "fromfile"

    def test_dotenv_malformed_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("this line has no equals\n")
        with pytest.raises(ValueError):
            load_settings(env={}, dotenv_path=env_file)


class TestConfigHelpers:
    def test_as_csv(self):
        assert _as_csv("") == []
        assert _as_csv("a,b , c") == ["a", "b", "c"]

    def test_load_dotenv_missing_file(self, tmp_path):
        assert _load_dotenv(tmp_path / "nope.env") == {}

    def test_load_dotenv_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=value\n")
        assert _load_dotenv(env_file) == {"KEY": "value"}


class TestLoadModels:
    def _write(self, tmp_path, content):
        path = tmp_path / "models.json"
        path.write_text(content)
        return path

    def test_valid(self, tmp_path):
        path = self._write(
            tmp_path,
            json.dumps({"models": [{"id": "human-gpt", "created": 5, "owned_by": "humangpt"}]}),
        )
        models = load_models(path)
        assert models == [{"id": "human-gpt", "object": "model", "created": 5, "owned_by": "humangpt"}]

    def test_defaults_applied(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"models": [{"id": "m"}]}))
        models = load_models(path)
        assert models[0]["created"] == 1715367049
        assert models[0]["owned_by"] == "humangpt"
        assert models[0]["object"] == "model"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(HumanGPTError):
            load_models(tmp_path / "missing.json")

    def test_bad_json_raises(self, tmp_path):
        path = self._write(tmp_path, "{not json")
        with pytest.raises(HumanGPTError):
            load_models(path)

    def test_bad_schema_raises(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"not_models": []}))
        with pytest.raises(HumanGPTError):
            load_models(path)

    def test_missing_id_raises(self, tmp_path):
        path = self._write(tmp_path, json.dumps({"models": [{"created": 1}]}))
        with pytest.raises(HumanGPTError):
            load_models(path)
