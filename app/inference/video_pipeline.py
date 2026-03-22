from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .runtime import get_video_runtime
from .schemas import CompressionItemResult, OutputFile, VideoOptions


@dataclass(slots=True)
class VideoClip:
    start: float
    end: float
    duration: float
    index: int


@dataclass(slots=True)
class VideoStats:
    duration: float
    fps: float
    total_frames: int
    has_audio: bool


def build_clip(start: float, end: float, index: int) -> VideoClip:
    start = max(0.0, float(start))
    end = max(start, float(end))
    return VideoClip(
        start=start,
        end=end,
        duration=max(0.0, end - start),
        index=index,
    )


def reindex_clips(clips: list[VideoClip]) -> list[VideoClip]:
    return [
        build_clip(clip.start, clip.end, index)
        for index, clip in enumerate(clips)
        if clip.duration > 1e-3
    ]


def resolve_video_mode(options: VideoOptions, request_fidelity: float) -> str:
    if options.mode is not None:
        return options.mode
    resolved_fidelity = options.fidelity if options.fidelity is not None else request_fidelity
    if resolved_fidelity < 0.4:
        return "aggressive"
    if resolved_fidelity < 0.7:
        return "balanced"
    return "conservative"


def resolve_video_fidelity(options: VideoOptions, request_fidelity: float) -> float:
    if options.fidelity is not None:
        return options.fidelity
    if options.mode is None:
        return request_fidelity
    mode = options.mode
    return {
        "conservative": 0.75,
        "balanced": 0.60,
        "aggressive": 0.40,
    }[mode]


def resolve_frame_count(duration: float) -> int:
    if duration < 3.0:
        return 1
    if duration <= 10.0:
        return 3
    return 5


def sample_positions(frame_count: int) -> list[float]:
    if frame_count <= 1:
        return [0.5]
    if frame_count == 3:
        return [0.2, 0.5, 0.8]
    if frame_count == 5:
        return [0.1, 0.3, 0.5, 0.7, 0.9]
    step = 1.0 / (frame_count + 1)
    return [step * (index + 1) for index in range(frame_count)]


def sample_timestamps_for_clip(clip: VideoClip) -> list[float]:
    positions = sample_positions(resolve_frame_count(clip.duration))
    if clip.duration <= 0:
        return [clip.start]
    return [
        min(clip.end - 1e-3, clip.start + (clip.duration * position))
        for position in positions
    ]


def parse_frame_rate(rate: str) -> float:
    if "/" in rate:
        numerator, denominator = rate.split("/", maxsplit=1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(rate)


def probe_video(video_path: Path) -> VideoStats:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,r_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("ffprobe is required for video compression.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(f"ffprobe failed for {video_path.name}: {exc.stderr.strip()}") from exc

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path.name}")

    duration = float(payload["format"]["duration"])
    fps = parse_frame_rate(video_stream.get("r_frame_rate", "0"))
    if duration <= 0:
        raise ValueError(f"Invalid video duration for {video_path.name}")
    if fps <= 0:
        raise ValueError(f"Invalid frame rate for {video_path.name}")

    return VideoStats(
        duration=duration,
        fps=fps,
        total_frames=max(1, int(round(duration * fps))),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def scenes_to_clips(scenes: Sequence[dict[str, Any]], stats: VideoStats) -> list[VideoClip]:
    if not scenes:
        return [build_clip(0.0, stats.duration, 0)]

    cursor = 0.0
    raw_clips: list[VideoClip] = []
    frame_duration = 1.0 / stats.fps

    for scene in scenes:
        start_frame = scene.get("start_frame")
        end_frame = scene.get("end_frame")

        if start_frame is not None:
            start = float(start_frame) / stats.fps
        else:
            start = float(scene.get("start_time", 0.0))

        if end_frame is not None:
            end = float(end_frame + 1) / stats.fps
        else:
            end = float(scene.get("end_time", stats.duration)) + frame_duration

        start = min(max(start, 0.0), stats.duration)
        end = min(max(end, start), stats.duration)

        if end <= start:
            continue

        if start > cursor + frame_duration:
            raw_clips.append(build_clip(cursor, start, len(raw_clips)))

        start = max(start, cursor)
        raw_clips.append(build_clip(start, end, len(raw_clips)))
        cursor = end

    if cursor < stats.duration:
        raw_clips.append(build_clip(cursor, stats.duration, len(raw_clips)))

    normalized = reindex_clips(raw_clips)
    if not normalized:
        return [build_clip(0.0, stats.duration, 0)]

    normalized[0] = build_clip(0.0, normalized[0].end, 0)
    normalized[-1] = build_clip(normalized[-1].start, stats.duration, len(normalized) - 1)
    return reindex_clips(normalized)


def merge_short_clips(
    clips: list[VideoClip],
    *,
    min_clip_seconds: float,
    total_duration: float,
) -> list[VideoClip]:
    merged = reindex_clips(clips)
    if len(merged) <= 1:
        return merged

    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, clip in enumerate(list(merged)):
            if clip.duration >= min_clip_seconds:
                continue

            if index == 0:
                neighbor = merged[1]
                merged[1] = build_clip(clip.start, neighbor.end, 1)
                del merged[0]
            elif index == len(merged) - 1:
                neighbor = merged[index - 1]
                merged[index - 1] = build_clip(neighbor.start, clip.end, index - 1)
                del merged[index]
            else:
                previous_clip = merged[index - 1]
                next_clip = merged[index + 1]
                if previous_clip.duration >= next_clip.duration:
                    merged[index - 1] = build_clip(previous_clip.start, clip.end, index - 1)
                    del merged[index]
                else:
                    merged[index + 1] = build_clip(clip.start, next_clip.end, index + 1)
                    del merged[index]

            merged = reindex_clips(merged)
            changed = True
            break

    if not merged:
        return [build_clip(0.0, total_duration, 0)]

    merged[0] = build_clip(0.0, merged[0].end, 0)
    merged[-1] = build_clip(merged[-1].start, total_duration, len(merged) - 1)
    return reindex_clips(merged)


def clip_center(clip: VideoClip) -> float:
    return clip.start + (clip.duration / 2.0)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(lhs * rhs for lhs, rhs in zip(left, right)))


def novelty_to_selected(
    index: int,
    selected_indices: Sequence[int],
    embeddings: Sequence[Sequence[float]],
) -> float:
    if not selected_indices:
        return 1.0
    max_similarity = max(cosine_similarity(embeddings[index], embeddings[selected]) for selected in selected_indices)
    return max(0.0, 1.0 - max_similarity)


def coverage_to_selected(
    index: int,
    selected_indices: Sequence[int],
    clips: Sequence[VideoClip],
    max_gap_seconds: float,
) -> float:
    if not selected_indices:
        return 1.0
    center = clip_center(clips[index])
    nearest_gap = min(abs(center - clip_center(clips[selected])) for selected in selected_indices)
    return min(1.0, nearest_gap / max_gap_seconds)


def select_task_agnostic_clips(
    clips: Sequence[VideoClip],
    embeddings: Sequence[Sequence[float]],
    *,
    target_fidelity: float,
    novelty_threshold: float,
    max_gap_seconds: float,
) -> list[int]:
    if not clips:
        return []
    if len(clips) == 1:
        return [0]

    total_duration = sum(clip.duration for clip in clips)
    target_duration = max(clips[0].duration, total_duration * target_fidelity)
    selected_indices = [0]
    selected_set = {0}
    kept_duration = clips[0].duration
    last_kept_index = 0

    for index in range(1, len(clips)):
        similarity = cosine_similarity(embeddings[index], embeddings[last_kept_index])
        time_since_last_kept = max(0.0, clips[index].start - clips[last_kept_index].end)
        should_keep = similarity < novelty_threshold or time_since_last_kept > max_gap_seconds
        if not should_keep:
            continue

        selected_indices.append(index)
        selected_set.add(index)
        kept_duration += clips[index].duration
        last_kept_index = index

    remaining = [index for index in range(len(clips)) if index not in selected_set]
    while remaining and kept_duration < target_duration:
        best_index = max(
            remaining,
            key=lambda index: (
                0.6 * novelty_to_selected(index, selected_indices, embeddings)
                + 0.4 * coverage_to_selected(index, selected_indices, clips, max_gap_seconds),
                -index,
            ),
        )
        selected_indices.append(best_index)
        selected_indices.sort()
        selected_set.add(best_index)
        kept_duration += clips[best_index].duration
        remaining.remove(best_index)

    return sorted(selected_indices)


def select_text_conditioned_clips(
    clips: Sequence[VideoClip],
    embeddings: Sequence[Sequence[float]],
    relevance_scores: Sequence[float],
    *,
    target_fidelity: float,
    max_gap_seconds: float,
) -> list[int]:
    if not clips:
        return []
    if len(clips) == 1:
        return [0]

    total_duration = sum(clip.duration for clip in clips)
    target_duration = max(clips[0].duration, total_duration * target_fidelity)
    selected_indices = [0]
    selected_set = {0}
    kept_duration = clips[0].duration
    remaining = [index for index in range(1, len(clips))]

    while remaining and kept_duration < target_duration:
        best_index = max(
            remaining,
            key=lambda index: (
                0.65 * max(0.0, min(1.0, (relevance_scores[index] + 1.0) / 2.0))
                + 0.25 * novelty_to_selected(index, selected_indices, embeddings)
                + 0.10 * coverage_to_selected(index, selected_indices, clips, max_gap_seconds),
                -index,
            ),
        )
        selected_indices.append(best_index)
        selected_indices.sort()
        selected_set.add(best_index)
        kept_duration += clips[best_index].duration
        remaining = [index for index in remaining if index != best_index]

    return sorted(selected_indices)


def apply_padding_and_merge(
    clips: Sequence[VideoClip],
    *,
    total_duration: float,
    padding_seconds: float,
    merge_gap_seconds: float,
) -> list[VideoClip]:
    segments: list[VideoClip] = []

    for clip in sorted(clips, key=lambda item: item.start):
        start = max(0.0, clip.start - padding_seconds)
        end = min(total_duration, clip.end + padding_seconds)

        if segments and start - segments[-1].end <= merge_gap_seconds:
            previous = segments[-1]
            segments[-1] = build_clip(previous.start, max(previous.end, end), previous.index)
            continue

        segments.append(build_clip(start, end, len(segments)))

    return reindex_clips(segments)


def extract_clip_frames(video_path: Path, clips: Sequence[VideoClip], stats: VideoStats) -> list[list[Any]]:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("OpenCV and Pillow are required for video frame sampling.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path.name}")

    clip_frames: list[list[Any]] = []
    try:
        for clip in clips:
            sampled_frames: list[Any] = []
            for timestamp in sample_timestamps_for_clip(clip):
                frame_index = min(stats.total_frames - 1, max(0, int(round(timestamp * stats.fps))))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(Image.fromarray(frame))

            if not sampled_frames:
                raise RuntimeError(f"Failed to sample frames for clip {clip.index} in {video_path.name}")

            clip_frames.append(sampled_frames)
    finally:
        capture.release()

    return clip_frames


def build_clip_embeddings(
    sampled_frames: Sequence[Sequence[Any]],
    *,
    prompt: str | None,
) -> tuple[list[list[float]], list[float]]:
    import torch

    runtime = get_video_runtime()
    flat_images: list[Any] = []
    counts_per_clip: list[int] = []
    for frames in sampled_frames:
        flat_images.extend(frames)
        counts_per_clip.append(len(frames))

    image_features: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(flat_images), 32):
            batch = flat_images[start : start + 32]
            inputs = runtime.clip_processor(images=batch, return_tensors="pt")
            tensor_inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
            features = runtime.clip_model.get_image_features(**tensor_inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            image_features.append(features.cpu())

        image_feature_tensor = torch.cat(image_features, dim=0)

        text_features = None
        if prompt:
            text_inputs = runtime.clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            tensor_inputs = {key: value.to(runtime.device) for key, value in text_inputs.items()}
            text_features = runtime.clip_model.get_text_features(**tensor_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features.cpu()

    clip_embeddings: list[list[float]] = []
    relevance_scores: list[float] = []
    offset = 0
    for frame_count in counts_per_clip:
        frame_tensor = image_feature_tensor[offset : offset + frame_count]
        offset += frame_count
        pooled = frame_tensor.mean(dim=0)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clip_embeddings.append(pooled.tolist())

        if text_features is None:
            relevance_scores.append(0.0)
            continue

        similarities = torch.matmul(frame_tensor, text_features[0])
        top_count = min(2, similarities.numel())
        relevance_scores.append(float(torch.topk(similarities, k=top_count).values.mean().item()))

    return clip_embeddings, relevance_scores


def build_filter_complex(segments: Sequence[VideoClip], *, has_audio: bool) -> tuple[str, str, str | None]:
    filters: list[str] = []
    for index, segment in enumerate(segments):
        filters.append(
            f"[0:v]trim=start={segment.start:.6f}:end={segment.end:.6f},setpts=PTS-STARTPTS[v{index}]"
        )
        if has_audio:
            filters.append(
                f"[0:a]atrim=start={segment.start:.6f}:end={segment.end:.6f},asetpts=PTS-STARTPTS[a{index}]"
            )

    if len(segments) == 1:
        return ";".join(filters), "[v0]", "[a0]" if has_audio else None

    if has_audio:
        concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(segments)))
        filters.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]")
        return ";".join(filters), "[outv]", "[outa]"

    concat_inputs = "".join(f"[v{index}]" for index in range(len(segments)))
    filters.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]")
    return ";".join(filters), "[outv]", None


def render_video_segments(
    *,
    input_path: Path,
    output_path: Path,
    segments: Sequence[VideoClip],
    has_audio: bool,
) -> None:
    filter_complex, video_label, audio_label = build_filter_complex(segments, has_audio=has_audio)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        video_label,
    ]
    if audio_label:
        command.extend(["-map", audio_label])

    command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"])

    if audio_label:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")

    command.extend(["-movflags", "+faststart", str(output_path)])

    try:
        subprocess.run(command, capture_output=True, check=True, text=True)
    except FileNotFoundError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("ffmpeg is required for video compression.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(f"ffmpeg failed for {input_path.name}: {exc.stderr.strip()}") from exc


def encode_output_file(source_name: str, data: bytes) -> OutputFile:
    source_stem = Path(source_name).stem or "video"
    return OutputFile(
        file_name=f"{source_stem}_compressed.mp4",
        content_type="video/mp4",
        data_base64=base64.b64encode(data).decode("ascii"),
        size_bytes=len(data),
    )


def compress_video_bytes(
    *,
    item_id: str,
    index: int,
    source_name: str,
    data: bytes,
    options: VideoOptions,
    request_fidelity: float,
) -> CompressionItemResult:
    if not data:
        raise ValueError(f"{source_name} is empty.")

    with tempfile.TemporaryDirectory(prefix="pied-piper-video-") as tmpdir:
        temp_dir = Path(tmpdir)
        input_path = temp_dir / Path(source_name).name
        input_path.write_bytes(data)

        stats = probe_video(input_path)
        runtime = get_video_runtime()
        scenes = runtime.scene_detector.detect_scenes(
            str(input_path),
            threshold=options.shot_threshold,
        )

        initial_clips = scenes_to_clips(scenes, stats)
        cleaned_clips = merge_short_clips(
            initial_clips,
            min_clip_seconds=options.min_clip_seconds,
            total_duration=stats.duration,
        )
        sampled_frames = extract_clip_frames(input_path, cleaned_clips, stats)
        clip_embeddings, relevance_scores = build_clip_embeddings(sampled_frames, prompt=options.prompt)

        resolved_video_fidelity = resolve_video_fidelity(options, request_fidelity)
        resolved_mode = resolve_video_mode(options, request_fidelity)

        if options.prompt:
            selected_indices = select_text_conditioned_clips(
                cleaned_clips,
                clip_embeddings,
                relevance_scores,
                target_fidelity=resolved_video_fidelity,
                max_gap_seconds=options.max_gap_seconds,
            )
        else:
            selected_indices = select_task_agnostic_clips(
                cleaned_clips,
                clip_embeddings,
                target_fidelity=resolved_video_fidelity,
                novelty_threshold=options.novelty_threshold,
                max_gap_seconds=options.max_gap_seconds,
            )

        if not selected_indices:
            selected_indices = [0]

        selected_clips = [cleaned_clips[selected_index] for selected_index in selected_indices]
        rendered_segments = apply_padding_and_merge(
            selected_clips,
            total_duration=stats.duration,
            padding_seconds=options.padding_seconds,
            merge_gap_seconds=options.merge_gap_seconds,
        )

        output_path = temp_dir / f"{Path(source_name).stem}_compressed.mp4"
        render_video_segments(
            input_path=input_path,
            output_path=output_path,
            segments=rendered_segments,
            has_audio=stats.has_audio,
        )

        output_bytes = output_path.read_bytes()
        if len(output_bytes) > options.max_inline_bytes:
            raise ValueError(
                f"Compressed video is {len(output_bytes)} bytes which exceeds the inline response "
                f"limit of {options.max_inline_bytes} bytes."
            )

        output_duration = sum(segment.duration for segment in rendered_segments)
        output_file = encode_output_file(source_name, output_bytes)

        return CompressionItemResult(
            id=item_id,
            index=index,
            modality="video",
            source_name=source_name,
            status="completed",
            output_file=output_file,
            message="Compressed video returned inline as an MP4 artifact.",
            metrics={
                "original_duration": round(stats.duration, 3),
                "output_duration": round(output_duration, 3),
                "reduction_ratio": round(max(0.0, 1.0 - (output_duration / stats.duration)), 4),
                "clips_total": len(cleaned_clips),
                "clips_kept": len(selected_clips),
                "clips_removed": max(0, len(cleaned_clips) - len(selected_clips)),
                "segments_rendered": len(rendered_segments),
                "request_fidelity": round(request_fidelity, 4),
                "resolved_video_fidelity": round(resolved_video_fidelity, 4),
                "resolved_mode": resolved_mode,
                "prompt_conditioned": bool(options.prompt),
                "fps": round(stats.fps, 3),
                "novelty_threshold": round(options.novelty_threshold, 4),
                "max_gap_seconds": round(options.max_gap_seconds, 3),
                "padding_seconds": round(options.padding_seconds, 3),
                "merge_gap_seconds": round(options.merge_gap_seconds, 3),
                "shot_threshold": round(options.shot_threshold, 3),
                "output_size_bytes": len(output_bytes),
                "kept_spans": [
                    {
                        "start": round(clip.start, 3),
                        "end": round(clip.end, 3),
                        "duration": round(clip.duration, 3),
                    }
                    for clip in selected_clips
                ],
                "rendered_spans": [
                    {
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "duration": round(segment.duration, 3),
                    }
                    for segment in rendered_segments
                ],
            },
        )
