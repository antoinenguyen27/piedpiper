from __future__ import annotations

from .schemas import CompressionItemResult


def stub_video(*, item_id: str, index: int, source_name: str) -> CompressionItemResult:
    return CompressionItemResult(
        id=item_id,
        index=index,
        modality="video",
        source_name=source_name,
        status="stubbed",
        message="Video compression is not implemented yet.",
    )

