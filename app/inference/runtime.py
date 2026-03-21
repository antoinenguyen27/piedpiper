from __future__ import annotations

from functools import lru_cache


MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


class MissingTextRuntimeError(RuntimeError):
    """Raised when the text compression runtime is unavailable."""


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
        device_map="cpu",
    )

