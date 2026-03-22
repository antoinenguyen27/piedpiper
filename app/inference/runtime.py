from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Pointing back to your custom BERT model
MODEL_NAME = "HuggingFacer112358/piedpiper"
MODEL_SUBFOLDER = "run_3/checkpoint-epoch-9"
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


class CustomBERTCompressor:
    """
    A drop-in replacement for LLMLingua that uses a custom BERT token classification
    model to compress text and properly stitch WordPiece subwords.
    """

    def __init__(self, model_name: str, subfolder: str, device: str):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise MissingTextRuntimeError(
                "Transformers and Torch are required."
            ) from exc

        self.device = device
        self.torch = torch

        # Pass the subfolder argument natively to Hugging Face
        kwargs = {"subfolder": subfolder} if subfolder else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, **kwargs
        ).to(device)
        self.model.eval()

    def compress_prompt(self, texts: list[str], rate: float, **kwargs) -> dict:
        compressed_prompts = []
        total_origin = 0
        total_compressed = 0

        for text in texts:
            # 1. Tokenize chunk (Safety truncation at 1024)
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)

            # DistilBERT does not accept token_type_ids; drop it if present
            inputs.pop("token_type_ids", None)

            input_ids = inputs["input_ids"][0]
            total_origin += len(input_ids)

            # 2. Inference
            with self.torch.no_grad():
                outputs = self.model(**inputs)

            # Assuming index 1 represents the "keep" class probabilities
            logits = outputs.logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)[:, 1]

            # 3. Apply target token rate
            target_token_count = max(1, int(len(input_ids) * rate))

            top_indices = self.torch.topk(probabilities, target_token_count).indices
            kept_indices = self.torch.sort(top_indices).values
            kept_ids = input_ids[kept_indices]

            # Drop special tokens ([CLS], [SEP])
            special_ids = set(self.tokenizer.all_special_ids)
            clean_kept_ids = [
                idx.item() for idx in kept_ids if idx.item() not in special_ids
            ]

            total_compressed += len(clean_kept_ids)

            # 4. The Stitcher: Decode back to clean text
            compressed_text = self.tokenizer.decode(
                clean_kept_ids, clean_up_tokenization_spaces=True
            )
            compressed_prompts.append(compressed_text)

        return {
            "compressed_prompt_list": compressed_prompts,
            "origin_tokens": total_origin,
            "compressed_tokens": total_compressed,
        }


@lru_cache(maxsize=1)
def get_prompt_compressor():
    device = _resolve_torch_device()
    try:
        return CustomBERTCompressor(
            model_name=MODEL_NAME, subfolder=MODEL_SUBFOLDER, device=device
        )
    except Exception as exc:
        raise MissingTextRuntimeError(
            f"Failed to load custom BERT compressor from {MODEL_NAME}/{MODEL_SUBFOLDER}. "
            "Ensure transformers and torch are installed."
        ) from exc


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
