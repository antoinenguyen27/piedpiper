from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
# MODEL_NAME = "HuggingFacer112358/piedpiper"
# MODEL_SUBFOLDER = "run_3/checkpoint-epoch-9"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class MissingTextRuntimeError(RuntimeError):
    """Raised when the text compression runtime is unavailable."""


class MissingVideoRuntimeError(RuntimeError):
    """Raised when the video compression runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class VideoRuntime:
    scene_detector: object
    clip_model: object
    clip_processor: object
    device: str


def _resolve_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_prompt_compressor():
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise MissingTextRuntimeError(
            "llmlingua is not installed. Add it to the inference runtime image."
        ) from exc

    return PromptCompressor(
        model_name=MODEL_NAME,
        use_llmlingua2=True,
        device_map=_resolve_torch_device(),
        # model_config={"subfolder": MODEL_SUBFOLDER},
    )


@lru_cache(maxsize=1)
def get_video_runtime() -> VideoRuntime:
    try:
        from transformers import AutoProcessor, CLIPModel
        from transnetv2_pytorch import TransNetV2
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise MissingVideoRuntimeError(
            "Video runtime dependencies are not installed. Add TransNetV2, Transformers, "
            "Torch, Pillow, ffmpeg-python, and OpenCV to the inference runtime image."
        ) from exc

    device = _resolve_torch_device()
    scene_detector = TransNetV2(device=device)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    clip_model.to(device)
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    return VideoRuntime(
        scene_detector=scene_detector,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
    )


def warm_inference_runtime() -> None:
    get_prompt_compressor()
    get_video_runtime()
