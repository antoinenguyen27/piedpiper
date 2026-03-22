from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextOptions(BaseModel):
    fidelity: float | None = Field(default=None, gt=0.0, lt=1.0)
    target_token: int = -1
    chunk_chars: int = 4000
    overlap_chars: int = 300
    drop_consecutive: bool = True


class VideoOptions(BaseModel):
    prompt: str | None = None
    fidelity: float | None = Field(default=None, gt=0.0, le=1.0)
    mode: Literal["conservative", "balanced", "aggressive"] | None = None
    novelty_threshold: float = Field(default=0.93, gt=0.0, lt=1.0)
    max_gap_seconds: float = Field(default=30.0, gt=0.0)
    min_clip_seconds: float = Field(default=1.0, gt=0.0)
    merge_gap_seconds: float = Field(default=0.75, ge=0.0)
    padding_seconds: float = Field(default=0.25, ge=0.0)
    shot_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    max_inline_bytes: int = Field(default=16_000_000, gt=0)


class RequestOptions(BaseModel):
    fidelity: float = Field(default=0.33, gt=0.0, lt=1.0)
    text: TextOptions = Field(default_factory=TextOptions)
    video: VideoOptions = Field(default_factory=VideoOptions)

    def resolved_text_fidelity(self) -> float:
        if self.text.fidelity is not None:
            return self.text.fidelity
        return self.fidelity

    def resolved_video_mode(self) -> Literal["conservative", "balanced", "aggressive"]:
        if self.video.mode is not None:
            return self.video.mode
        if self.fidelity < 0.4:
            return "conservative"
        if self.fidelity < 0.7:
            return "balanced"
        return "aggressive"

    def resolved_video_fidelity(self) -> float:
        if self.video.fidelity is not None:
            return self.video.fidelity
        mode = self.resolved_video_mode()
        return {
            "conservative": 0.75,
            "balanced": 0.60,
            "aggressive": 0.40,
        }[mode]


class ManifestItem(BaseModel):
    id: str
    index: int
    source_type: Literal["inline_text", "upload"]
    source_name: str
    text: str | None = None
    upload_field: str | None = None
    content_type: str | None = None


class CompressionManifest(BaseModel):
    sdk_version: str = "0.2.0"
    options: RequestOptions = Field(default_factory=RequestOptions)
    items: list[ManifestItem]


class OutputFile(BaseModel):
    file_name: str
    content_type: str
    data_base64: str
    size_bytes: int


class CompressionItemResult(BaseModel):
    id: str
    index: int
    modality: Literal["text", "image", "video"]
    source_name: str
    status: Literal["completed", "passthrough", "stubbed", "failed"]
    output_text: str | None = None
    output_file: OutputFile | None = None
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
