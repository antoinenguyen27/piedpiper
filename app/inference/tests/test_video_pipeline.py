from __future__ import annotations

import pytest

from app.inference.schemas import VideoOptions
from app.inference.video_pipeline import (
    VideoClip,
    apply_padding_and_merge,
    merge_short_clips,
    resolve_video_fidelity,
    resolve_video_mode,
    select_task_agnostic_clips,
)


def test_resolve_video_fidelity_uses_request_fidelity_directly_by_default():
    options = VideoOptions()
    assert resolve_video_fidelity(options, 0.9) == 0.9


def test_resolve_video_fidelity_uses_low_request_fidelity_directly_by_default():
    options = VideoOptions()
    assert resolve_video_fidelity(options, 0.33) == 0.33


def test_resolve_video_fidelity_respects_explicit_mode_override():
    options = VideoOptions(mode="balanced")
    assert resolve_video_fidelity(options, 0.9) == 0.60


def test_resolve_video_mode_buckets_direct_fidelity_for_metrics():
    options = VideoOptions()
    assert resolve_video_mode(options, 0.9) == "conservative"
    assert resolve_video_mode(options, 0.55) == "balanced"
    assert resolve_video_mode(options, 0.2) == "aggressive"


def test_video_options_reject_conflicting_fidelity_and_mode():
    with pytest.raises(ValueError):
        VideoOptions(fidelity=0.8, mode="balanced")


def test_merge_short_clips_absorbs_fragmentation():
    clips = [
        VideoClip(start=0.0, end=5.0, duration=5.0, index=0),
        VideoClip(start=5.0, end=5.4, duration=0.4, index=1),
        VideoClip(start=5.4, end=10.0, duration=4.6, index=2),
    ]

    merged = merge_short_clips(clips, min_clip_seconds=1.0, total_duration=10.0)

    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[-1].end == 10.0


def test_select_task_agnostic_clips_uses_novelty_and_budget():
    clips = [
        VideoClip(start=0.0, end=4.0, duration=4.0, index=0),
        VideoClip(start=4.0, end=8.0, duration=4.0, index=1),
        VideoClip(start=8.0, end=12.0, duration=4.0, index=2),
        VideoClip(start=12.0, end=16.0, duration=4.0, index=3),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    selected = select_task_agnostic_clips(
        clips,
        embeddings,
        target_fidelity=0.75,
        novelty_threshold=0.93,
        max_gap_seconds=30.0,
    )

    assert selected == [0, 2, 3]


def test_apply_padding_and_merge_coalesces_small_gaps():
    clips = [
        VideoClip(start=2.0, end=4.0, duration=2.0, index=0),
        VideoClip(start=4.3, end=6.0, duration=1.7, index=1),
    ]

    merged = apply_padding_and_merge(
        clips,
        total_duration=10.0,
        padding_seconds=0.25,
        merge_gap_seconds=0.75,
    )

    assert len(merged) == 1
    assert round(merged[0].start, 2) == 1.75
    assert round(merged[0].end, 2) == 6.25
