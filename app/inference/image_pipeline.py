from __future__ import annotations

from .schemas import CompressionItemResult


def passthrough_image(*, item_id: str, index: int, source_name: str) -> CompressionItemResult:
    return CompressionItemResult(
        id=item_id,
        index=index,
        modality="image",
        source_name=source_name,
        status="passthrough",
        message="Image inputs are accepted as passthrough in phase 0.",
    )

