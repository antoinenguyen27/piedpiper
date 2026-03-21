from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile, status

from .image_pipeline import passthrough_image
from .schemas import CompressionManifest, CompressionItemResult, CompressionResponse, UsageSummary
from .text_pipeline import SourceText, TEXT_SUFFIXES, compress_text_sources, extract_text, normalize_text
from .video_pipeline import stub_video

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


async def parse_manifest_request(request: Request) -> tuple[CompressionManifest, dict[str, UploadFile]]:
    form = await request.form()
    manifest_raw = form.get("manifest")
    if not isinstance(manifest_raw, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing manifest form field.",
        )

    try:
        manifest = CompressionManifest.model_validate(json.loads(manifest_raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid manifest: {exc}",
        ) from exc

    uploads: dict[str, UploadFile] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            uploads[key] = value

    return manifest, uploads


def classify_modality(source_name: str, *, source_type: str) -> str:
    if source_type == "inline_text":
        return "text"

    suffix = Path(source_name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"

    raise ValueError(f"Unsupported file type: {source_name}")


async def process_manifest(
    manifest: CompressionManifest,
    uploads: dict[str, UploadFile],
    *,
    request_id: str,
) -> CompressionResponse:
    results: list[CompressionItemResult] = []
    pending_text: list[SourceText] = []

    for item in sorted(manifest.items, key=lambda entry: entry.index):
        try:
            modality = classify_modality(item.source_name, source_type=item.source_type)

            if item.source_type == "inline_text":
                if item.text is None:
                    raise ValueError("Inline text item is missing text.")
                cleaned = normalize_text(item.text)
                if not cleaned:
                    raise ValueError("Inline text item is empty after normalization.")
                pending_text.append(
                    SourceText(
                        item_id=item.id,
                        index=item.index,
                        source_name=item.source_name,
                        text=cleaned,
                    )
                )
                continue

            upload_field = item.upload_field
            if not upload_field:
                raise ValueError("Upload item is missing upload_field.")

            upload = uploads.get(upload_field)
            if upload is None:
                raise ValueError(f"Missing uploaded file for field {upload_field}.")

            if modality == "text":
                data = await upload.read()
                extracted = extract_text(item.source_name, data)
                pending_text.append(
                    SourceText(
                        item_id=item.id,
                        index=item.index,
                        source_name=item.source_name,
                        text=extracted,
                    )
                )
                continue

            if modality == "image":
                results.append(
                    passthrough_image(
                        item_id=item.id,
                        index=item.index,
                        source_name=item.source_name,
                    )
                )
                continue

            if modality == "video":
                results.append(
                    stub_video(
                        item_id=item.id,
                        index=item.index,
                        source_name=item.source_name,
                    )
                )
                continue

            raise ValueError(f"Unsupported modality for {item.source_name}.")
        except Exception as exc:
            try:
                failed_modality = classify_modality(item.source_name, source_type=item.source_type)
            except Exception:
                failed_modality = "text"
            results.append(
                CompressionItemResult(
                    id=item.id,
                    index=item.index,
                    modality=failed_modality,
                    source_name=item.source_name,
                    status="failed",
                    error=str(exc),
                )
            )

    usage = UsageSummary()
    if pending_text:
        try:
            text_results, text_usage = compress_text_sources(pending_text, manifest.options.text)
            results.extend(text_results)
            usage = UsageSummary(**text_usage)
        except Exception as exc:
            failed_ids = {result.id for result in results}
            for source in pending_text:
                if source.item_id in failed_ids:
                    continue
                results.append(
                    CompressionItemResult(
                        id=source.item_id,
                        index=source.index,
                        modality="text",
                        source_name=source.source_name,
                        status="failed",
                        error=str(exc),
                    )
                )

    ordered_results = sorted(results, key=lambda entry: entry.index)
    statuses = {result.status for result in ordered_results}
    if not ordered_results or statuses == {"failed"}:
        top_level_status = "failed"
    elif "failed" in statuses:
        top_level_status = "partial_success"
    else:
        top_level_status = "completed"

    return CompressionResponse(
        request_id=request_id,
        status=top_level_status,
        items=ordered_results,
        usage=usage,
    )
