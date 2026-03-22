from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .exceptions import RequestError

TEXT_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".pptx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES | VIDEO_SUFFIXES


@dataclass(slots=True)
class FileUpload:
    field_name: str
    path: Path
    content_type: str


@dataclass(slots=True)
class NormalizedRequest:
    manifest: dict[str, Any]
    uploads: list[FileUpload]


def _flatten_input(input_value: Any) -> list[Any]:
    if isinstance(input_value, (str, Path)):
        return [input_value]
    if isinstance(input_value, Sequence):
        flattened: list[Any] = []
        for entry in input_value:
            flattened.extend(_flatten_input(entry))
        return flattened
    return [input_value]


def _is_ambiguous_file_string(value: str) -> bool:
    try:
        return Path(value).expanduser().exists()
    except (OSError, RuntimeError, ValueError):
        return False


def _validate_fidelity(fidelity: float) -> float:
    value = float(fidelity)
    if not 0.0 < value < 1.0:
        raise RequestError("fidelity must be between 0 and 1.")
    return value


def normalize_input(input_value: Any, *, fidelity: float = 0.33) -> NormalizedRequest:
    fidelity = _validate_fidelity(fidelity)
    items = _flatten_input(input_value)
    manifest_items: list[dict[str, Any]] = []
    uploads: list[FileUpload] = []

    for index, item in enumerate(items):
        item_id = f"item_{index}"

        if isinstance(item, str):
            if _is_ambiguous_file_string(item):
                raise RequestError(
                    "Plain strings are treated as inline text. Use pathlib.Path for file inputs."
                )
            manifest_items.append(
                {
                    "id": item_id,
                    "index": index,
                    "source_type": "inline_text",
                    "source_name": f"raw_text_{index}",
                    "text": item,
                }
            )
            continue

        if isinstance(item, Path):
            path = item.expanduser()
        elif hasattr(item, "__fspath__"):
            path = Path(item)
        else:
            raise RequestError(f"Unsupported input type: {type(item)!r}")

        if not path.exists():
            raise RequestError(f"Input file does not exist: {path}")
        if path.is_dir():
            raise RequestError(f"Directories are not supported: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise RequestError(
                f"Unsupported file type: {suffix}. Supported types: {sorted(SUPPORTED_SUFFIXES)}"
            )

        field_name = f"file_{len(uploads)}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        manifest_items.append(
            {
                "id": item_id,
                "index": index,
                "source_type": "upload",
                "source_name": path.name,
                "upload_field": field_name,
                "content_type": content_type,
            }
        )
        uploads.append(FileUpload(field_name=field_name, path=path, content_type=content_type))

    return NormalizedRequest(
        manifest={
            "sdk_version": "0.1.0",
            "options": {
                "fidelity": fidelity,
                "text": {
                    "fidelity": fidelity,
                    "target_token": -1,
                    "chunk_chars": 4000,
                    "overlap_chars": 300,
                    "drop_consecutive": True,
                },
                "video": {
                    "prompt": None,
                    "fidelity": None,
                    "mode": None,
                    "novelty_threshold": 0.93,
                    "max_gap_seconds": 30.0,
                    "min_clip_seconds": 1.0,
                    "merge_gap_seconds": 0.75,
                    "padding_seconds": 0.25,
                    "shot_threshold": 0.5,
                    "max_inline_bytes": 16000000,
                }
            },
            "items": manifest_items,
        },
        uploads=uploads,
    )
