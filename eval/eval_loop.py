from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PIED_PIPER_SRC = REPO_ROOT / "app" / "packaging" / "src"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_PROVIDER = "openai"
DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"
DEFAULT_SPLIT = "test"
DEFAULT_VALIDATION_SPLIT = "validation"
DEFAULT_FIDELITY = 0.9
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 6
DEFAULT_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_BACKOFF_CAP_SECONDS = 120.0
DEFAULT_JITTER_MAX_SECONDS = 0.75
DEFAULT_COMPRESSION_TIMEOUT_SECONDS = 600.0
DEFAULT_OPENAI_TIMEOUT_SECONDS = 600.0
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
ANSWER_ALPHABET = "ABCDEFGHIJ"
AnswerLetter = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

OPENAI_INSTRUCTIONS = (
    "You are an expert multiple-choice test taker solving MMLU-Pro questions. "
    "Use the worked examples in the input as reference, think step by step, keep "
    "the reasoning concise, and return JSON that matches the schema exactly. "
    "Set `answer` to the single uppercase option letter for the final question. "
    "Only choose from options that appear in the final question."
)


class EvalError(RuntimeError):
    """Base eval error."""


class MissingDependencyError(EvalError):
    """Raised when an optional runtime dependency is unavailable."""


@dataclass(slots=True)
class EvalConfig:
    provider: str = DEFAULT_PROVIDER
    model: str | None = None
    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    num_samples: int = 50
    seed: int = 42
    fidelity: float = DEFAULT_FIDELITY
    use_compression: bool = True
    concurrency: int = DEFAULT_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS
    jitter_max_seconds: float = DEFAULT_JITTER_MAX_SECONDS
    compression_timeout_seconds: float = DEFAULT_COMPRESSION_TIMEOUT_SECONDS
    openai_timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_name: str | None = None
    resume: bool = True
    save_config: bool = True

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "openai":
            return DEFAULT_OPENAI_MODEL
        raise EvalError(f"Unsupported provider: {self.provider!r}")

    @property
    def output_stem(self) -> str:
        if self.run_name:
            return slugify(self.run_name)

        compression_tag = "compressed" if self.use_compression else "raw"
        fidelity_tag = f"f{self.fidelity:.2f}".replace(".", "p")
        model_tag = slugify(self.resolved_model)
        return (
            f"{slugify(self.dataset_name)}_{slugify(self.split)}_"
            f"{slugify(self.provider)}_{model_tag}_{compression_tag}_"
            f"{fidelity_tag}_n{self.num_samples}_seed{self.seed}"
        )

    @property
    def results_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_summary.json"

    @property
    def config_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_config.json"


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    dataset_index: int
    question: str
    options: list[str]
    answer: str
    category: str
    cot_content: str | None
    raw_item: dict[str, Any]


@dataclass(slots=True)
class RetryOutcome:
    value: Any
    attempts: int
    elapsed_seconds: float


@dataclass(slots=True)
class EvalDataset:
    samples: list[EvalSample]
    validation_by_category: dict[str, list[EvalSample]]


@dataclass(slots=True)
class PromptBundle:
    text: str
    exemplar_ids: list[str]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro evals with optional Pied Piper compression."
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fidelity", type=float, default=DEFAULT_FIDELITY)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=DEFAULT_BACKOFF_BASE_SECONDS,
    )
    parser.add_argument(
        "--backoff-cap-seconds",
        type=float,
        default=DEFAULT_BACKOFF_CAP_SECONDS,
    )
    parser.add_argument(
        "--jitter-max-seconds",
        type=float,
        default=DEFAULT_JITTER_MAX_SECONDS,
    )
    parser.add_argument(
        "--compression-timeout-seconds",
        type=float,
        default=DEFAULT_COMPRESSION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-compression",
        dest="use_compression",
        action="store_false",
        help="Skip Pied Piper compression and send raw prompts to the model.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from an existing JSONL results file.",
    )
    parser.add_argument(
        "--no-save-config",
        dest="save_config",
        action="store_false",
        help="Skip writing the run configuration JSON file.",
    )
    parser.set_defaults(use_compression=True, resume=True, save_config=True)

    args = parser.parse_args()
    config = EvalConfig(
        provider=args.provider,
        model=args.model,
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        fidelity=args.fidelity,
        use_compression=args.use_compression,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_cap_seconds=args.backoff_cap_seconds,
        jitter_max_seconds=args.jitter_max_seconds,
        compression_timeout_seconds=args.compression_timeout_seconds,
        openai_timeout_seconds=args.openai_timeout_seconds,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        save_config=args.save_config,
    )
    validate_config(config)
    return config


def validate_config(config: EvalConfig) -> None:
    if config.provider != "openai":
        raise EvalError(
            "Unsupported provider. This eval runner currently supports "
            "provider='openai'."
        )
    if not 0.0 < config.fidelity < 1.0:
        raise EvalError("fidelity must be between 0 and 1.")
    if config.num_samples <= 0:
        raise EvalError("num_samples must be greater than 0.")
    if config.concurrency <= 0:
        raise EvalError("concurrency must be greater than 0.")
    if config.max_retries <= 0:
        raise EvalError("max_retries must be greater than 0.")
    if config.backoff_base_seconds <= 0.0:
        raise EvalError("backoff_base_seconds must be greater than 0.")
    if config.backoff_cap_seconds < config.backoff_base_seconds:
        raise EvalError(
            "backoff_cap_seconds must be greater than or equal to backoff_base_seconds."
        )
    if config.jitter_max_seconds < 0.0:
        raise EvalError("jitter_max_seconds must be non-negative.")


def import_module_or_raise(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"Missing dependency {module_name!r}. Install it with: {install_hint}"
        ) from exc


def import_pied_piper_module() -> Any:
    if LOCAL_PIED_PIPER_SRC.is_dir():
        sdk_src = str(LOCAL_PIED_PIPER_SRC)
        if sdk_src not in sys.path:
            sys.path.insert(0, sdk_src)
    return import_module_or_raise(
        "pied_piper",
        "python3 -m pip install -e app/packaging",
    )


def create_openai_client(config: EvalConfig) -> Any:
    openai = import_module_or_raise(
        "openai",
        "python3 -m pip install openai",
    )
    try:
        client = openai.AsyncOpenAI(timeout=config.openai_timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        raise EvalError(
            "OpenAI client configuration failed. Set OPENAI_API_KEY or pass a valid "
            "OpenAI environment before running this eval."
        ) from exc
    if not hasattr(client.responses, "parse"):
        raise MissingDependencyError(
            "Installed OpenAI SDK does not support responses.parse. "
            "Upgrade it with: python3 -m pip install -U openai"
        )
    return client


def load_mmlu_dataset(config: EvalConfig) -> EvalDataset:
    datasets = import_module_or_raise(
        "datasets",
        "python3 -m pip install datasets",
    )
    dataset = datasets.load_dataset(config.dataset_name)
    if config.split not in dataset:
        available_splits = ", ".join(sorted(dataset.keys()))
        raise EvalError(
            f"Split {config.split!r} is unavailable for {config.dataset_name!r}. "
            f"Available splits: {available_splits}"
        )
    if DEFAULT_VALIDATION_SPLIT not in dataset:
        available_splits = ", ".join(sorted(dataset.keys()))
        raise EvalError(
            f"Validation split {DEFAULT_VALIDATION_SPLIT!r} is unavailable for "
            f"{config.dataset_name!r}. Available splits: {available_splits}"
        )

    test_split = dataset[config.split].shuffle(seed=config.seed)
    total_rows = len(test_split)
    selected_rows = min(config.num_samples, total_rows)
    test_split = test_split.select(range(selected_rows))

    samples: list[EvalSample] = []
    for dataset_index, item in enumerate(test_split):
        samples.append(
            normalize_mmlu_item(
                item,
                dataset_index,
                split_name=config.split,
            )
        )

    validation_by_category: dict[str, list[EvalSample]] = {}
    for dataset_index, item in enumerate(dataset[DEFAULT_VALIDATION_SPLIT]):
        normalized = normalize_mmlu_item(
            item,
            dataset_index,
            split_name=DEFAULT_VALIDATION_SPLIT,
        )
        validation_by_category.setdefault(normalized.category, []).append(normalized)

    if not validation_by_category:
        raise EvalError(
            f"No validation exemplars were loaded from {config.dataset_name!r}."
        )

    return EvalDataset(
        samples=samples,
        validation_by_category=validation_by_category,
    )


def normalize_mmlu_item(
    item: dict[str, Any],
    dataset_index: int,
    *,
    split_name: str,
) -> EvalSample:
    question = str(item.get("question", "")).strip()
    if not question:
        raise EvalError(
            f"Dataset item at index {dataset_index} in split {split_name!r} "
            "is missing a question."
        )

    category = str(item.get("category", "")).strip()
    if not category:
        raise EvalError(
            f"Dataset item at index {dataset_index} in split {split_name!r} "
            "is missing a category."
        )

    raw_options = item.get("options", [])
    options = [str(option).strip() for option in raw_options if str(option).strip()]
    options = [option for option in options if option.upper() != "N/A"]
    if not options:
        raise EvalError(
            f"Dataset item at index {dataset_index} in split {split_name!r} "
            "has no usable options."
        )

    raw_answer = item.get("answer")
    if raw_answer in (None, ""):
        raw_answer_index = item.get("answer_index")
        if isinstance(raw_answer_index, int) and 0 <= raw_answer_index < len(options):
            answer = chr(65 + raw_answer_index)
        else:
            raise EvalError(
                f"Dataset item at index {dataset_index} in split {split_name!r} "
                "is missing a valid answer."
            )
    else:
        answer = normalize_answer(raw_answer)

    answer_index = ord(answer) - 65
    if answer_index >= len(options):
        raise EvalError(
            f"Answer {answer!r} is out of range for {len(options)} option(s) "
            f"at dataset index {dataset_index} in split {split_name!r}."
        )
    sample_id = str(
        item.get("question_id")
        or item.get("id")
        or item.get("uuid")
        or f"{split_name}_{dataset_index}"
    )

    return EvalSample(
        sample_id=sample_id,
        dataset_index=dataset_index,
        question=question,
        options=options,
        answer=answer,
        category=category,
        cot_content=str(item.get("cot_content", "")).strip() or None,
        raw_item=dict(item),
    )


def normalize_answer(raw_answer: Any) -> str:
    if isinstance(raw_answer, int):
        if raw_answer < 0:
            raise EvalError(f"Unsupported negative answer index: {raw_answer}")
        return chr(65 + raw_answer)

    answer_text = str(raw_answer).strip().upper()
    if answer_text.isdigit():
        return chr(65 + int(answer_text))

    match = re.search(r"[A-J]", answer_text)
    if not match:
        raise EvalError(f"Unsupported answer value: {raw_answer!r}")
    return match.group(0)


def format_mmlu_example(sample: EvalSample, cot_content: str | None) -> str:
    example = f"Question: {sample.question}\nOptions: "
    for index, option in enumerate(sample.options):
        example += f"{ANSWER_ALPHABET[index]}. {option}\n"

    if cot_content is None:
        return f"{example}Answer: "

    normalized_cot = cot_content.strip()
    if normalized_cot.startswith("A: "):
        normalized_cot = normalized_cot[3:]
    return f"{example}Answer: {normalized_cot}\n\n"


def build_mmlu_prompt(
    sample: EvalSample,
    validation_by_category: dict[str, list[EvalSample]],
) -> PromptBundle:
    exemplars = validation_by_category.get(sample.category)
    if not exemplars:
        raise EvalError(
            f"No validation exemplars found for category {sample.category!r}."
        )

    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{sample.category}.\n\n"
    )
    exemplar_ids: list[str] = []
    for exemplar in exemplars:
        prompt += format_mmlu_example(
            exemplar,
            exemplar.cot_content or "Let's think step by step.",
        )
        exemplar_ids.append(exemplar.sample_id)

    prompt += format_mmlu_example(sample, "Let's think step by step.")
    return PromptBundle(text=prompt, exemplar_ids=exemplar_ids)


def extract_answer_letter(response_text: str) -> tuple[str | None, str | None]:
    text = response_text.strip()
    if not text:
        return None, None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "answer" in payload:
        try:
            return normalize_answer(payload["answer"]), "json"
        except EvalError:
            pass

    json_matches = re.findall(r'"answer"\s*:\s*"([A-J])"', text, re.IGNORECASE)
    if json_matches:
        return json_matches[-1].upper(), "json_regex"

    answer_matches = re.findall(
        r"the\s+answer\s+is\s*\(?([A-J])\)?",
        text,
        re.IGNORECASE,
    )
    if answer_matches:
        return answer_matches[-1].upper(), "answer_is"

    line_matches = re.findall(
        r"^\s*answer\s*[:\-]?\s*\(?([A-J])\)?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if line_matches:
        return line_matches[-1].upper(), "answer_line"

    return None, None


@lru_cache(maxsize=1)
def get_openai_response_model() -> type[Any]:
    pydantic = import_module_or_raise(
        "pydantic",
        "python3 -m pip install pydantic",
    )
    BaseModel = pydantic.BaseModel
    ConfigDict = pydantic.ConfigDict
    Field = pydantic.Field

    class OpenAIEvalAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        reasoning: str | None = Field(
            default=None,
            description="Brief step-by-step reasoning for the final question.",
        )
        answer: AnswerLetter = Field(
            description="Single uppercase option letter for the final answer.",
        )

    return OpenAIEvalAnswer


def is_retryable_pied_piper_error(exc: Exception) -> bool:
    module_name = type(exc).__module__
    class_name = type(exc).__name__
    if module_name.startswith("pied_piper"):
        if class_name == "ServerError":
            return True
        if class_name in {"AuthenticationError", "ConfigurationError"}:
            return False

    message = str(exc).lower()
    retryable_fragments = [
        "429",
        "408",
        "409",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
    ]
    return any(fragment in message for fragment in retryable_fragments)


def is_retryable_openai_error(exc: Exception) -> bool:
    if type(exc).__module__ != "openai":
        return False

    class_name = type(exc).__name__
    if class_name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}:
        return True

    if class_name == "APIStatusError":
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        return status_code in RETRYABLE_STATUS_CODES

    return False


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    value = headers.get("retry-after")
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, retry_at.timestamp() - time.time())


def compute_backoff_seconds(
    *,
    attempt: int,
    base_seconds: float,
    cap_seconds: float,
    jitter_max_seconds: float,
    retry_after_seconds: float | None,
) -> float:
    if retry_after_seconds is not None:
        floor_seconds = retry_after_seconds
    else:
        floor_seconds = min(cap_seconds, base_seconds * math.pow(2, attempt - 1))
    return floor_seconds + random.uniform(0.0, jitter_max_seconds)


async def maybe_jitter(config: EvalConfig) -> None:
    if config.jitter_max_seconds <= 0.0:
        return
    await asyncio.sleep(random.uniform(0.0, config.jitter_max_seconds))


async def run_with_retry(
    *,
    label: str,
    config: EvalConfig,
    call: Any,
    is_retryable: Any,
    retry_after_getter: Any | None = None,
) -> RetryOutcome:
    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        await maybe_jitter(config)
        try:
            value = await call()
            return RetryOutcome(
                value=value,
                attempts=attempt,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not is_retryable(exc) or attempt >= config.max_retries:
                break

            retry_after_seconds = None
            if retry_after_getter is not None:
                retry_after_seconds = retry_after_getter(exc)

            delay_seconds = compute_backoff_seconds(
                attempt=attempt,
                base_seconds=config.backoff_base_seconds,
                cap_seconds=config.backoff_cap_seconds,
                jitter_max_seconds=config.jitter_max_seconds,
                retry_after_seconds=retry_after_seconds,
            )
            print(
                f"[retry] {label} attempt {attempt}/{config.max_retries} failed: "
                f"{exc}. Sleeping {delay_seconds:.2f}s before retry."
            )
            await asyncio.sleep(delay_seconds)

    assert last_error is not None
    raise last_error


def compress_with_pied_piper_sync(
    prompt: str,
    fidelity: float,
    timeout_seconds: float = DEFAULT_COMPRESSION_TIMEOUT_SECONDS,
) -> Any:
    pied_piper = import_pied_piper_module()
    client = pied_piper.Client(timeout=timeout_seconds)
    try:
        return client.compress(prompt, fidelity=fidelity)
    finally:
        client.close()


async def compress_prompt(prompt: str, config: EvalConfig) -> tuple[str, dict[str, Any]]:
    if not config.use_compression:
        return prompt, {
            "enabled": False,
            "status": "skipped",
            "requested_fidelity": config.fidelity,
            "attempts": 0,
            "request_id": None,
            "item_status": None,
            "metrics": None,
            "usage": None,
        }

    async def do_compress() -> Any:
        return await asyncio.to_thread(
            compress_with_pied_piper_sync,
            prompt,
            config.fidelity,
            config.compression_timeout_seconds,
        )

    outcome = await run_with_retry(
        label="pied_piper.compress",
        config=config,
        call=do_compress,
        is_retryable=is_retryable_pied_piper_error,
    )

    result = outcome.value
    result_text = getattr(result, "text", None) or prompt
    item = result.items[0] if getattr(result, "items", None) else None
    metadata = {
        "enabled": True,
        "status": getattr(result, "status", None),
        "requested_fidelity": config.fidelity,
        "attempts": outcome.attempts,
        "request_id": getattr(result, "request_id", None),
        "item_status": getattr(item, "status", None) if item else None,
        "metrics": to_jsonable(getattr(item, "metrics", None) if item else None),
        "usage": to_jsonable(getattr(result, "usage", None)),
        "elapsed_seconds": round(outcome.elapsed_seconds, 3),
    }
    return result_text, metadata


async def generate_openai_response(
    prompt: str,
    config: EvalConfig,
    client: Any,
) -> tuple[str, str | None, dict[str, Any]]:
    response_model = get_openai_response_model()

    async def do_request() -> Any:
        return await client.responses.parse(
            model=config.resolved_model,
            instructions=OPENAI_INSTRUCTIONS,
            input=prompt,
            text_format=response_model,
            max_output_tokens=384,
            reasoning={"effort": DEFAULT_REASONING_EFFORT},
            temperature=0.0,
        )

    outcome = await run_with_retry(
        label=f"{config.provider}.responses.parse",
        config=config,
        call=do_request,
        is_retryable=is_retryable_openai_error,
        retry_after_getter=extract_retry_after_seconds,
    )

    response = outcome.value
    structured_output = getattr(response, "output_parsed", None)
    structured_answer = None
    if structured_output is not None:
        raw_structured_answer = getattr(structured_output, "answer", None)
        if raw_structured_answer is not None:
            structured_answer = normalize_answer(raw_structured_answer)

    output_text = (getattr(response, "output_text", None) or "").strip()
    if not output_text and structured_output is not None:
        output_text = json.dumps(
            to_jsonable(structured_output),
            ensure_ascii=True,
        )
    elif not output_text:
        output_text = str(response)

    metadata = {
        "provider": config.provider,
        "model": config.resolved_model,
        "attempts": outcome.attempts,
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "requested_reasoning_effort": DEFAULT_REASONING_EFFORT,
        "error": to_jsonable(getattr(response, "error", None)),
        "incomplete_details": to_jsonable(getattr(response, "incomplete_details", None)),
        "structured_output": to_jsonable(structured_output),
        "usage": to_jsonable(getattr(response, "usage", None)),
        "elapsed_seconds": round(outcome.elapsed_seconds, 3),
    }
    return output_text, structured_answer, metadata


async def run_sample(
    sample: EvalSample,
    validation_by_category: dict[str, list[EvalSample]],
    config: EvalConfig,
    openai_client: Any,
) -> dict[str, Any]:
    prompt_bundle = build_mmlu_prompt(sample, validation_by_category)
    raw_prompt = prompt_bundle.text
    started = time.perf_counter()

    compressed_prompt, compression_metadata = await compress_prompt(raw_prompt, config)
    response_text, structured_answer, provider_metadata = await generate_openai_response(
        compressed_prompt,
        config,
        openai_client,
    )

    predicted = structured_answer
    prediction_source = "structured_output" if structured_answer is not None else None
    if predicted is None:
        predicted, prediction_source = extract_answer_letter(response_text)
    correct = predicted == sample.answer if predicted is not None else False

    return {
        "sample_id": sample.sample_id,
        "dataset_index": sample.dataset_index,
        "question_id": sample.raw_item.get("question_id"),
        "category": sample.category,
        "answer": sample.answer,
        "predicted": predicted,
        "prediction_source": prediction_source,
        "correct": correct,
        "status": "completed",
        "few_shot_examples": len(prompt_bundle.exemplar_ids),
        "few_shot_example_ids": prompt_bundle.exemplar_ids,
        "raw_prompt_length": len(raw_prompt),
        "final_prompt_length": len(compressed_prompt),
        "response_text": response_text,
        "provider": provider_metadata,
        "compression": compression_metadata,
        "timing": {
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "updated_at": utc_now_iso(),
    }


def read_existing_results(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    ordered_records: list[dict[str, Any]] = []
    if not path.exists():
        return records_by_id, ordered_records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            records_by_id[sample_id] = record
            ordered_records.append(record)
    return records_by_id, ordered_records


async def append_jsonl(path: Path, lock: asyncio.Lock, record: dict[str, Any]) -> None:
    async with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_summary(
    *,
    config: EvalConfig,
    samples: list[EvalSample],
    records_by_id: dict[str, dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    records = [records_by_id[sample.sample_id] for sample in samples if sample.sample_id in records_by_id]
    completed_records = [record for record in records if record.get("status") == "completed"]
    error_records = [record for record in records if record.get("status") == "error"]
    correct_count = sum(1 for record in completed_records if record.get("correct"))
    parsed_count = sum(1 for record in completed_records if record.get("predicted") is not None)
    accuracy = correct_count / len(completed_records) if completed_records else 0.0

    return {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "totals": {
            "requested_samples": config.num_samples,
            "selected_samples": len(samples),
            "completed": len(completed_records),
            "errors": len(error_records),
            "parsed_predictions": parsed_count,
            "correct": correct_count,
            "accuracy": accuracy,
        },
        "paths": {
            "results_jsonl": str(config.results_path),
            "summary_json": str(config.summary_path),
            "config_json": str(config.config_path),
        },
    }


async def evaluate(config: EvalConfig) -> dict[str, Any]:
    started_at = utc_now_iso()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_config:
        config_payload = {**asdict(config), "output_dir": str(config.output_dir)}
        config.config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    dataset = load_mmlu_dataset(config)
    samples = dataset.samples
    records_by_id, _ = read_existing_results(config.results_path)
    completed_ids = set(records_by_id)

    pending_samples = [sample for sample in samples if sample.sample_id not in completed_ids]
    if completed_ids and config.resume:
        print(
            f"Resuming from {config.results_path}. "
            f"Skipping {len(completed_ids)} existing sample(s)."
        )
    elif config.results_path.exists() and not config.resume:
        raise EvalError(
            f"Results file already exists at {config.results_path}. "
            "Pass --no-resume only with a fresh run-name or remove the existing file."
        )

    if not pending_samples:
        summary = build_summary(
            config=config,
            samples=samples,
            records_by_id=records_by_id,
            started_at=started_at,
        )
        config.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    openai_client = create_openai_client(config)
    try:
        semaphore = asyncio.Semaphore(config.concurrency)
        write_lock = asyncio.Lock()

        async def worker(sample: EvalSample) -> dict[str, Any]:
            async with semaphore:
                try:
                    record = await run_sample(
                        sample,
                        dataset.validation_by_category,
                        config,
                        openai_client,
                    )
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "sample_id": sample.sample_id,
                        "dataset_index": sample.dataset_index,
                        "question_id": sample.raw_item.get("question_id"),
                        "category": sample.category,
                        "answer": sample.answer,
                        "predicted": None,
                        "correct": False,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "updated_at": utc_now_iso(),
                    }
                await append_jsonl(config.results_path, write_lock, record)
                return record

        tasks = [asyncio.create_task(worker(sample)) for sample in pending_samples]

        progress = None
        try:
            progress = get_progress_bar(total=len(tasks), desc="Evaluating")
            for completed_task in asyncio.as_completed(tasks):
                record = await completed_task
                records_by_id[record["sample_id"]] = record
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        summary = build_summary(
            config=config,
            samples=samples,
            records_by_id=records_by_id,
            started_at=started_at,
        )
        config.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        totals = summary["totals"]
        print("=" * 40)
        print("EVALUATION COMPLETE")
        print(f"Provider: {config.provider}")
        print(f"Model:    {config.resolved_model}")
        print(f"Split:    {config.split}")
        print(f"Accuracy: {totals['accuracy']:.2%}")
        print(f"Correct:  {totals['correct']}/{totals['completed']}")
        print(f"Errors:   {totals['errors']}")
        print(f"Results:  {config.results_path}")
        print(f"Summary:  {config.summary_path}")
        print("=" * 40)
        return summary
    finally:
        await openai_client.close()


def get_progress_bar(total: int, desc: str) -> Any:
    if total <= 0:
        return None
    try:
        tqdm = import_module_or_raise("tqdm", "python3 -m pip install tqdm")
    except MissingDependencyError:
        return None
    return tqdm.tqdm(total=total, desc=desc)


def print_dependency_snapshot() -> None:
    dependencies = {
        "openai": importlib_util.find_spec("openai") is not None,
        "datasets": importlib_util.find_spec("datasets") is not None,
        "pydantic": importlib_util.find_spec("pydantic") is not None,
        "tqdm": importlib_util.find_spec("tqdm") is not None,
    }
    if LOCAL_PIED_PIPER_SRC.is_dir():
        dependencies["pied_piper(repo-local)"] = True
    else:
        dependencies["pied_piper(repo-local)"] = False
        dependencies["pied_piper(installed)"] = (
            importlib_util.find_spec("pied_piper") is not None
        )

    print("Dependency snapshot:")
    for name, available in dependencies.items():
        print(f"  - {name}: {'available' if available else 'missing'}")


def main() -> int:
    try:
        config = parse_args()
        print_dependency_snapshot()
        asyncio.run(evaluate(config))
        return 0
    except (EvalError, MissingDependencyError) as exc:
        print(f"Eval setup error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
