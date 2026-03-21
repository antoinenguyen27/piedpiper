from __future__ import annotations

from pathlib import Path

import pytest

from pied_piper.exceptions import RequestError
from pied_piper.inputs import normalize_input


def test_normalize_inline_text():
    request = normalize_input("hello")
    assert request.manifest["items"][0]["source_type"] == "inline_text"
    assert request.manifest["items"][0]["text"] == "hello"


def test_normalize_path_input(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    request = normalize_input(path)
    item = request.manifest["items"][0]
    assert item["source_type"] == "upload"
    assert item["source_name"] == "note.txt"
    assert request.uploads[0].field_name == "file_0"


def test_reject_plain_string_path(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(RequestError):
        normalize_input(str(path))


def test_reject_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(RequestError):
        normalize_input(path)

