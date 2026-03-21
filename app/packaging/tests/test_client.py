from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pied_piper import Client
from pied_piper.exceptions import AuthenticationError


def test_client_sends_auth_and_manifest(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        body = request.read().decode("utf-8", errors="ignore")
        assert "name=\"manifest\"" in body
        assert "note.txt" in body
        return httpx.Response(
            200,
            json={
                "request_id": "req_123",
                "status": "completed",
                "items": [
                    {
                        "id": "item_0",
                        "index": 0,
                        "modality": "text",
                        "source_name": "note.txt",
                        "status": "completed",
                        "output_text": "compressed",
                    }
                ],
                "usage": {"origin_tokens": 10, "compressed_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    with Client(
        base_url="https://example.test",
        api_key="secret",
        http_client=httpx.Client(base_url="https://example.test", transport=transport),
    ) as client:
        result = client.compress(path)

    assert result.status == "completed"
    assert result.text == "compressed"


def test_client_maps_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "nope"})

    transport = httpx.MockTransport(handler)
    with Client(
        base_url="https://example.test",
        api_key="secret",
        http_client=httpx.Client(base_url="https://example.test", transport=transport),
    ) as client:
        with pytest.raises(AuthenticationError):
            client.compress("hello")
