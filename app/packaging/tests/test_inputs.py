from __future__ import annotations

from pathlib import Path

import pytest

from pied_piper.exceptions import RequestError
from pied_piper.inputs import normalize_input


def test_normalize_inline_text():
    request = normalize_input("hello", fidelity=0.55)
    assert request.manifest["items"][0]["source_type"] == "inline_text"
    assert request.manifest["items"][0]["text"] == "hello"
    assert request.manifest["options"]["fidelity"] == 0.55
    assert request.manifest["options"]["text"]["fidelity"] == 0.55


def test_normalize_inline_text_uses_default_fidelity():
    request = normalize_input("hello")
    assert request.manifest["options"]["fidelity"] == 0.9
    assert request.manifest["options"]["text"]["fidelity"] == 0.9


def test_normalize_multiline_inline_text():
    text = """
Pied Piper is acting as a preprocessing step before the LLM call.
The goal is to shorten verbose source material while keeping the main facts.
This toy example uses only inline text so the setup stays simple.
The compressed result is then forwarded into a normal OpenAI API request.
""".strip()

    request = normalize_input(text)
    assert request.manifest["items"][0]["source_type"] == "inline_text"
    assert request.manifest["items"][0]["text"] == text


def test_normalize_inline_text_when_path_probe_raises_oserror(monkeypatch: pytest.MonkeyPatch):
    def raise_oserror(self) -> bool:
        raise OSError("file name too long")

    monkeypatch.setattr(Path, "exists", raise_oserror)

    request = normalize_input("inline text that should not be treated as a path")
    assert request.manifest["items"][0]["source_type"] == "inline_text"


def test_normalize_inline_text_when_expanduser_raises(monkeypatch: pytest.MonkeyPatch):
    def raise_runtime_error(self) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", raise_runtime_error)

    request = normalize_input("~/not-a-real-user/note.txt")
    assert request.manifest["items"][0]["source_type"] == "inline_text"


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


def test_reject_invalid_fidelity():
    with pytest.raises(RequestError):
        normalize_input("hello", fidelity=1.5)
