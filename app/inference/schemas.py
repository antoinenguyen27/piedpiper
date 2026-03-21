from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextOptions(BaseModel):
    rate: float = 0.33
    target_token: int = -1
    chunk_chars: int = 4000
    overlap_chars: int = 300


class RequestOptions(BaseModel):
    text: TextOptions = Field(default_factory=TextOptions)


class ManifestItem(BaseModel):
    id: str
    index: int
    source_type: Literal["inline_text", "upload"]
    source_name: str
    text: str | None = None
    upload_field: str | None = None
    content_type: str | None = None


class CompressionManifest(BaseModel):
    sdk_version: str = "0.1.0"
    options: RequestOptions = Field(default_factory=RequestOptions)
    items: list[ManifestItem]


class CompressionItemResult(BaseModel):
    id: str
    index: int
    modality: Literal["text", "image", "video"]
    source_name: str
    status: Literal["completed", "passthrough", "stubbed", "failed"]
    output_text: str | None = None
    message: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


class UsageSummary(BaseModel):
    origin_tokens: int = 0
    compressed_tokens: int = 0


class CompressionResponse(BaseModel):
    request_id: str
    status: Literal["completed", "partial_success", "failed"]
    items: list[CompressionItemResult]
    usage: UsageSummary | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "pied-piper-inference"
    auth_configured: bool
    text_backend: str
    scale_to_zero: bool = True

