from __future__ import annotations

import io
import json
import os

from fastapi.testclient import TestClient

from app.inference.api import create_app


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-key"}


def test_health_and_root(monkeypatch):
    monkeypatch.setenv("PIED_PIPER_API_KEY", "test-key")
    client = TestClient(create_app())

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "pied-piper-inference"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["auth_configured"] is True


def test_compress_mixed_inputs(monkeypatch):
    monkeypatch.setenv("PIED_PIPER_API_KEY", "test-key")
    client = TestClient(create_app())

    def fake_compress_text_sources(sources, options, *, request_fidelity):
        assert request_fidelity == 0.55
        return (
            [
                {
                    "id": source.item_id,
                    "index": source.index,
                    "modality": "text",
                    "source_name": source.source_name,
                    "status": "completed",
                    "output_text": f"compressed:{source.text}",
                    "metrics": {"origin_tokens": 10, "compressed_tokens": 5, "compression_rate": 0.5},
                }
                for source in sources
            ],
            {"origin_tokens": 10 * len(sources), "compressed_tokens": 5 * len(sources)},
        )

    def fake_compress_video_bytes(*, item_id, index, source_name, data, options, request_fidelity):
        assert source_name == "video.mp4"
        assert data == b"video-bytes"
        assert request_fidelity == 0.55
        return {
            "id": item_id,
            "index": index,
            "modality": "video",
            "source_name": source_name,
            "status": "completed",
            "output_file": {
                "file_name": "video_compressed.mp4",
                "content_type": "video/mp4",
                "data_base64": "dmlkZW8=",
                "size_bytes": 5,
            },
            "metrics": {"clips_kept": 2, "clips_removed": 1},
        }

    monkeypatch.setattr("app.inference.router.compress_text_sources", fake_compress_text_sources)
    monkeypatch.setattr("app.inference.router.compress_video_bytes", fake_compress_video_bytes)

    manifest = {
        "options": {"fidelity": 0.55},
        "items": [
            {
                "id": "item_0",
                "index": 0,
                "source_type": "inline_text",
                "source_name": "raw_text_0",
                "text": "Hello world",
            },
            {
                "id": "item_1",
                "index": 1,
                "source_type": "upload",
                "source_name": "image.png",
                "upload_field": "file_0",
                "content_type": "image/png",
            },
            {
                "id": "item_2",
                "index": 2,
                "source_type": "upload",
                "source_name": "video.mp4",
                "upload_field": "file_1",
                "content_type": "video/mp4",
            },
        ]
    }

    response = client.post(
        "/v1/compress",
        headers=_auth_headers(),
        data={"manifest": json.dumps(manifest)},
        files=[
            ("file_0", ("image.png", io.BytesIO(b"png-bytes"), "image/png")),
            ("file_1", ("video.mp4", io.BytesIO(b"video-bytes"), "video/mp4")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert [item["status"] for item in payload["items"]] == ["completed", "passthrough", "completed"]
    assert payload["items"][2]["output_file"]["content_type"] == "video/mp4"


def test_auth_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("PIED_PIPER_API_KEY", "test-key")
    client = TestClient(create_app())

    response = client.post("/v1/compress", data={"manifest": json.dumps({"items": []})})
    assert response.status_code == 401
