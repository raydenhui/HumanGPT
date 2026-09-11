"""Unit tests for the pure helpers in humangpt/live_stream.py.

Importing live_stream pulls in models/db but no network — safe for unit tests.
"""

from __future__ import annotations

from humangpt.live_stream import _missing_suffix


def test_missing_suffix_complete():
    assert _missing_suffix("hello", "hello") == ""


def test_missing_suffix_partial_prefix():
    assert _missing_suffix("hello", "hello world") == " world"


def test_missing_suffix_from_empty():
    assert _missing_suffix("", "hello") == "hello"


def test_missing_suffix_diverged_is_safe():
    # emitted is not a prefix of full -> don't re-send (avoid duplication)
    assert _missing_suffix("world", "hello world") == ""


def test_missing_suffix_leading_separator_tail():
    # realistic word-chunk case: streamed "1 2 3 4", answered "1 2 3 4 5"
    assert _missing_suffix("1 2 3 4", "1 2 3 4 5") == " 5"
