from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class CompressionItemResult:
    id: str
    index: int
    modality: Literal["text", "image", "video"]
    source_name: str
    status: Literal["completed", "passthrough", "stubbed", "failed"]
    output_text: str | None = None
    message: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class CompressionResult:
    request_id: str
    status: Literal["completed", "partial_success", "failed"]
    items: list[CompressionItemResult] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        outputs = [
            item.output_text
            for item in sorted(self.items, key=lambda entry: entry.index)
            if item.status == "completed" and item.output_text
        ]
        return "\n\n".join(outputs)


def compression_result_from_dict(payload: dict[str, Any]) -> CompressionResult:
    items = [CompressionItemResult(**item) for item in payload.get("items", [])]
    return CompressionResult(
        request_id=payload["request_id"],
        status=payload["status"],
        items=items,
        usage=payload.get("usage"),
    )

