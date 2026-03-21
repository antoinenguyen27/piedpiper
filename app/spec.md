# Pied Piper App Specification

Status: draft for implementation  
Date: 2026-03-21  
Scope: `app/` only. The repository root remains model training and experimentation.

## 1. Context

This repository currently mixes two different concerns:

- training and experimentation at the repository root
- the hackathon product, which is a Python SDK plus a Modal-hosted inference service

The current codebase already gives us two important anchors:

- text prototype logic exists in `experimetal-token-compression/inference.py`
- video code at the repo root is still training-oriented (`train.py`, `stream_processing.py`, `cnn_compress.py`)

That means the product plan should be:

- make text inference real first
- accept image inputs as passthrough from day one
- implement video context compression as a pre-inference clip selector that returns an inline MP4 artifact plus metadata

The `app/` directory is therefore the product surface. The repository root stays as training code.

## 2. Goals

- Create a clean `app/` application area without disturbing the training root.
- Define a Python SDK that is pip-installable and exposes a top-level one-liner:
  - `pied_piper.compress(input)`
- Keep the SDK lightweight. It is an HTTP client, not a local inference runtime.
- Put modality routing, document extraction, batching, and compression logic in the Modal service.
- Use a single shared API key for the hackathon, stored as a Modal Secret and exposed to the service via environment variables.
- Keep the first public contract stable while extending it to carry inline binary artifacts for video outputs.

## 3. Non-goals

- Multi-tenant auth, per-user API keys, JWT issuance, OAuth, billing, or usage quotas.
- Local ML inference in the SDK.
- Repackaging the current training root as a publishable package.
- General file URL ingestion, directory ingestion, or in-memory PIL/numpy image ingestion in phase 0.

## 4. High-level Product Shape

The product is split into two independently understandable parts:

1. `app/packaging`
   - a publishable Python SDK
   - lightweight dependency set
   - handles input normalization, auth headers, HTTP transport, response parsing, and developer ergonomics

2. `app/inference`
   - a Modal application serving FastAPI over HTTP
   - owns auth verification, modality detection, extraction, batching, and inference orchestration
   - eventually loads both text and video models into one GPU-backed runtime

## 5. Proposed Repository Layout

```text
app/
  spec.md
  inference/
    modal_app.py
    api.py
    auth.py
    schemas.py
    router.py
    text_pipeline.py
    video_pipeline.py
    image_pipeline.py
    runtime.py
  packaging/
    README.md
    pyproject.toml
    src/
      pied_piper/
        __init__.py
        client.py
        config.py
        inputs.py
        models.py
        exceptions.py
        py.typed
    tests/
      test_inputs.py
      test_client.py
```

Notes:

- `app/inference` is not a publishable SDK package in phase 0. Its dependency source of truth will be the Modal image declaration in code.
- `app/packaging` is the only publishable Python package in phase 0.

## 6. Naming

### 6.1 Import name

The import package name is:

- `pied_piper`

This gives the desired developer experience:

```python
import pied_piper

result = pied_piper.compress("hello world")
```

### 6.2 Distribution name

The PyPI distribution name should **not** assume that `pied-piper` or `pied_piper` will be available.

Reason:

- PyPI normalizes hyphens and underscores for project names
- the `pied-piper` namespace is not cleanly available based on web search alone
- we should avoid spending time on a publish-time rename if the normalized namespace is already occupied or ambiguous

Recommended initial distribution name:

- `piedpiper-sdk`

This keeps the clean import name while avoiding likely namespace collision and brand ambiguity.

Final PyPI availability still needs to be confirmed at publish time.

## 7. SDK Public Contract

### 7.1 Public API surface

The SDK will expose:

```python
import pied_piper

result = pied_piper.compress(input)
```

Under the hood, this is a convenience wrapper around a configurable client:

```python
from pied_piper import Client

client = Client()
result = client.compress(input)
```

### 7.2 Public objects

The package should export:

- `compress`
- `Client`
- `CompressionResult`
- `CompressionItemResult`
- `PiedPiperError`
- `ConfigurationError`
- `AuthenticationError`
- `RequestError`

### 7.3 Input contract

The SDK should support these input shapes:

| Python input | Meaning | Current behavior |
| --- | --- | --- |
| `str` | raw inline text | compressed |
| `os.PathLike[str]` / `pathlib.Path` to `.txt`, `.md`, `.pdf`, `.docx`, `.pptx` | text-bearing file | uploaded, extracted remotely, compressed |
| `os.PathLike[str]` / `pathlib.Path` to image suffix | image file | uploaded or represented, returned as passthrough |
| `os.PathLike[str]` / `pathlib.Path` to video suffix | video file | uploaded, compressed into an inline MP4 artifact |
| sequence of supported inputs | mixed batch | handled in original order |

Current rules:

- `str` means raw text, not file path.
- file inputs must be `PathLike`, not plain strings.
- directories are rejected.
- raw `bytes`, URLs, file-like objects, and glob patterns are out of scope for phase 0.
- `fidelity` is the shared SDK parameter and always means the same thing: higher values preserve more for both text and video.

This is deliberate. Treating plain strings as possible paths creates ambiguous behavior and a brittle API.

### 7.4 Supported suffixes

Initial recognized suffixes:

- text: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`
- image: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`
- video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

Text note:

- `.md` is not handled in the current experimental text file, but it should be added in the productized inference path and treated as UTF-8 text.

### 7.5 Result contract

The SDK should return a structured object, not a bare string.

Recommended model:

```python
CompressionResult(
    request_id: str,
    status: Literal["completed", "partial_success", "failed"],
    items: list[CompressionItemResult],
    usage: UsageSummary | None,
)
```

```python
CompressionItemResult(
    id: str,
    index: int,
    modality: Literal["text", "image", "video"],
    source_name: str,
    status: Literal["completed", "passthrough", "stubbed", "failed"],
    output_text: str | None = None,
    output_file: OutputFile | None = None,
    message: str | None = None,
    metrics: dict[str, object] | None = None,
    error: str | None = None,
)
```

Behavior:

- text items use `status="completed"` and fill `output_text`
- image items use `status="passthrough"`
- video items use `status="completed"` and fill `output_file` with an inline MP4 artifact
- failed item extraction/compression uses `status="failed"` with `error`

Top-level status semantics:

- `completed`: no item failed
- `partial_success`: one or more items failed, but at least one succeeded or was accepted as passthrough/stubbed
- `failed`: request-level failure, auth failure, validation failure, or every item failed

Convenience:

- `CompressionResult.text` should concatenate completed text outputs in original input order
- non-text items are omitted from the `.text` convenience view

## 8. SDK Configuration

### 8.1 Environment variables

The SDK should read configuration from environment variables by default:

- `PIED_PIPER_BASE_URL`
- `PIED_PIPER_API_KEY`

Optional timeout handling can be constructor-based rather than environment-based in phase 0.

### 8.2 Constructor contract

```python
Client(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | httpx.Timeout | None = None,
)
```

Resolution order:

1. explicit argument
2. environment variable
3. raise `ConfigurationError`

### 8.3 Timeout policy

The SDK should not rely on HTTPX defaults alone.

Reason:

- HTTPX defaults to a five-second inactivity timeout
- larger uploads and cold-started inference can legitimately exceed that

Recommended default timeout profile:

- connect: 10s
- write: 60s
- read: 300s
- pool: 10s

This is a design recommendation based on expected request shape, not a library requirement.

## 9. SDK Internal Structure

### 9.1 `src/pied_piper/__init__.py`

Responsibilities:

- export the public API
- implement module-level `compress(...)`
- lazily create a default `Client` using environment configuration

### 9.2 `src/pied_piper/client.py`

Responsibilities:

- own an `httpx.Client`
- build auth headers
- build multipart requests
- deserialize JSON responses into SDK models
- translate transport and HTTP errors into SDK exceptions

### 9.3 `src/pied_piper/inputs.py`

Responsibilities:

- normalize user input into a flat ordered list
- validate file suffixes and file existence
- classify input as inline text or upload
- assign stable per-item ids and indices
- build request manifest entries

### 9.4 `src/pied_piper/models.py`

Responsibilities:

- define result models
- provide convenience helpers like `.text`

These can be `dataclass`-based in phase 0 to keep the SDK dependency footprint small.

### 9.5 `src/pied_piper/exceptions.py`

Recommended exception hierarchy:

- `PiedPiperError`
- `ConfigurationError`
- `AuthenticationError`
- `RequestError`
- `ServerError`

## 10. SDK Request Construction

The SDK should always call the backend using `multipart/form-data`, even for text-only requests.

Reason:

- FastAPI supports `File` and `Form` together
- FastAPI cannot simultaneously receive files/form data and a JSON body in the same path operation
- a single multipart contract keeps the server and client simple and uniform

### 10.1 Request form fields

The client should send:

- form field `manifest`
- zero or more file parts named `file_0`, `file_1`, `file_2`, ...

### 10.2 Manifest shape

Recommended manifest:

```json
{
  "sdk_version": "0.1.0",
  "options": {
    "text": {
      "rate": 0.33,
      "target_token": -1,
      "chunk_chars": 4000,
      "overlap_chars": 300
    }
  },
  "items": [
    {
      "id": "item_0",
      "index": 0,
      "source_type": "inline_text",
      "source_name": "raw_text_0",
      "text": "hello world"
    },
    {
      "id": "item_1",
      "index": 1,
      "source_type": "upload",
      "source_name": "paper.pdf",
      "upload_field": "file_0",
      "content_type": "application/pdf"
    }
  ]
}
```

Design rules:

- the manifest preserves original input ordering
- upload items reference multipart file parts by `upload_field`
- the server remains the authority on modality routing
- `content_type` is advisory metadata, not trusted classification

### 10.3 HTTP headers

The SDK should send:

- `Authorization: Bearer <PIED_PIPER_API_KEY>`

The client should not use Modal-specific proxy auth headers in phase 0.

### 10.4 HTTP transport details

Use `httpx.Client`, not module-level one-off request helpers.

Reason:

- connection pooling
- clearer lifecycle management
- easier testability

Files must be opened in binary mode.

## 11. Inference Service Contract

### 11.1 Endpoint surface

The inference service should expose:

- `GET /`
- `GET /health`
- `POST /v1/compress`

Phase 0 expectations:

- `GET /` may return a simple hello-world payload for smoke testing
- `GET /health` returns service metadata and readiness
- `POST /v1/compress` returns the final request/response contract, including inline video artifacts

This avoids writing the SDK twice.

### 11.2 Request body

`POST /v1/compress` accepts `multipart/form-data` only.

Implementation model:

- parse the incoming multipart form from `Request`
- read `manifest` from the form data
- collect `UploadFile` entries referenced by manifest `upload_field` values

`python-multipart` is required by FastAPI for forms/files.

### 11.3 Response body

The service returns JSON shaped like the SDK result contract. Example:

```json
{
  "request_id": "req_123",
  "status": "completed",
  "items": [
    {
      "id": "item_0",
      "index": 0,
      "modality": "text",
      "source_name": "raw_text_0",
      "status": "completed",
      "output_text": "compressed text here",
      "metrics": {
        "origin_tokens": 100,
        "compressed_tokens": 33,
        "compression_rate": 0.33
      }
    },
    {
      "id": "item_1",
      "index": 1,
      "modality": "video",
      "source_name": "demo.mp4",
      "status": "completed",
      "output_file": {
        "file_name": "demo_compressed.mp4",
        "content_type": "video/mp4",
        "data_base64": "...",
        "size_bytes": 123456
      },
      "metrics": {
        "original_duration": 12.5,
        "output_duration": 7.8,
        "clips_total": 5,
        "clips_kept": 3
      }
    }
  ],
  "usage": {
    "origin_tokens": 100,
    "compressed_tokens": 33
  }
}
```

## 12. Authentication Model

### 12.1 Hackathon auth decision

Hackathon auth is a single shared bearer token.

Source of truth:

- Modal Secret containing `PIED_PIPER_API_KEY=<shared secret>`

The Modal app injects the secret into the container environment.

### 12.2 Why this auth design

We are intentionally **not** using Modal proxy auth in phase 0.

Reason:

- the SDK should speak standard HTTP bearer auth
- the auth contract should remain stable if infrastructure changes later
- backend-side bearer validation is enough for the hackathon

### 12.3 Backend auth behavior

The FastAPI app should:

1. read the `Authorization` header
2. require `Bearer <token>`
3. compare the token against `os.environ["PIED_PIPER_API_KEY"]`
4. return `401 Unauthorized` on missing or invalid token

Comparison should be constant-time.

### 12.4 SDK auth behavior

The SDK should:

- read `PIED_PIPER_API_KEY`
- send `Authorization: Bearer <token>`
- raise `AuthenticationError` on 401

## 13. Modal Deployment Design

### 13.1 Why `@modal.asgi_app`

The inference service should use `@modal.asgi_app`, not `@modal.fastapi_endpoint`.

Reason:

- `fastapi_endpoint` is for simple single-handler endpoints
- we need multiple routes: `/`, `/health`, `/v1/compress`
- Modal docs explicitly recommend `@modal.asgi_app` for user-defined FastAPI applications with multiple routes

### 13.2 GPU deployment

When text and video inference are both active:

- move to a single GPU-backed runtime
- load both text and video runtimes once per container start
- mount a shared model cache for Hugging Face artifacts
- keep `min_containers=0` so the service still scales to zero

Initial concurrency stance:

- do **not** enable request concurrency for the GPU runtime until memory and thread-safety are measured
- process one request per container initially

This is the simplest robust path for a single-GPU service.

### 13.3 GPU choice

The final GPU type is intentionally left open in this spec.

Reason:

- the service must hold both the text and video runtimes on one GPU
- the cheapest listed Modal GPU may not fit both models

Practical rule:

- choose the lowest-cost GPU that can actually hold both runtimes with operational headroom

## 14. Inference Internal Structure

### 14.1 `app/inference/modal_app.py`

Responsibilities:

- define the Modal `App`
- define the base `Image`
- attach the Modal Secret
- expose the FastAPI app with `@modal.asgi_app`

### 14.2 `app/inference/api.py`

Responsibilities:

- create the FastAPI app
- wire routes
- register auth dependency
- call router/orchestrator functions

### 14.3 `app/inference/auth.py`

Responsibilities:

- parse bearer token
- validate against environment variable
- raise FastAPI `HTTPException(401)`

### 14.4 `app/inference/schemas.py`

Responsibilities:

- request and response models for `manifest`, item results, and health payloads

Pydantic is appropriate here because the server is already coupled to FastAPI.

### 14.5 `app/inference/router.py`

Responsibilities:

- parse manifest
- bind uploaded files to manifest entries
- detect modality
- partition items into text/image/video groups
- reassemble final results in original order

### 14.6 `app/inference/runtime.py`

Responsibilities:

- model loading and lifecycle-managed runtime ownership
- shared LLMLingua-2, TransNetV2, and CLIP accessors

## 15. Modality Routing Rules

### 15.1 Text

Text modality includes:

- inline raw text
- `.txt`
- `.md`
- extracted text from `.pdf`, `.docx`, `.pptx`

Text routing steps:

1. extract raw text when needed
2. normalize text
3. split into chunks
4. batch across text units
5. run text compressor
6. aggregate results per original input item

### 15.2 Image

Images are accepted in phase 0 but are not compressed.

Behavior:

- return `status="passthrough"`
- keep metadata such as file name and modality
- do not pretend the image was compressed

### 15.3 Video

Videos are compressed before downstream inference using shot segmentation, CLIP scoring, and budgeted clip selection.

Behavior:

- return `status="completed"` when compression succeeds
- include an inline MP4 artifact in `output_file`
- include clip-selection metadata in `metrics`

This keeps the API contract stable while still making the compressed video directly usable.

## 16. Text Pipeline Productization Plan

The experimental text logic in `experimetal-token-compression/inference.py` should be migrated into `app/inference` and treated as source material, not imported directly from the training root.

Reason:

- the root remains experimentation/training
- the current directory naming and layout are not product-grade
- the application surface should be self-contained

Logic to carry over:

- text normalization
- file extraction for `.txt`, `.pdf`, `.docx`, `.pptx`
- paragraph-aware chunking with overlap
- adaptive text batching
- LLMLingua compressor invocation

Productization changes:

- add `.md` support
- separate extraction/chunking/batching into smaller functions
- preserve item ordering and per-item metadata
- convert all print-based output into structured response objects
- convert failures into item-level errors whenever possible

## 17. Video Pipeline

The application video path now implements a pre-downstream-inference clip selector:

- TransNetV2 shot segmentation
- clip cleanup for short fragments
- 1/3/5 frame sampling by clip duration
- CLIP ViT-B/32 image embeddings
- budgeted clip selection with task-agnostic novelty/coverage scoring
- optional prompt-conditioned scoring
- FFmpeg stitching back into a compressed MP4 artifact

The Modal runtime loads LLMLingua-2, TransNetV2, and CLIP into the same container image, with shared model cache storage.

## 18. Packaging Specification

### 18.1 Packaging layout

The SDK subproject lives entirely under `app/packaging`.

Directory shape:

```text
app/packaging/
  README.md
  pyproject.toml
  src/
    pied_piper/
      __init__.py
      client.py
      config.py
      inputs.py
      models.py
      exceptions.py
      py.typed
  tests/
```

### 18.2 Build backend choice

Use `setuptools` with `pyproject.toml`.

Reason:

- standard, widely understood, and well-documented
- works cleanly with `src/` layout
- does not require a legacy `setup.py` executable workflow

### 18.3 `pyproject.toml`

Recommended initial contents:

```toml
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "piedpiper-sdk"
version = "0.1.0"
description = "Python SDK for Pied Piper multimodal compression"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [
  { name = "Pied Piper" }
]
dependencies = [
  "httpx>=0.27,<1.0"
]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Software Development :: Libraries :: Python Modules"
]

[project.optional-dependencies]
dev = [
  "build>=1.2",
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.6"
]

[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["pied_piper*"]
namespaces = false

[tool.setuptools.package-data]
pied_piper = ["py.typed"]
```

Notes:

- license should match the repo root license, which is Apache-2.0
- a subproject-local `README.md` should exist because packaging is built from `app/packaging`, not from the repository root

### 18.4 README requirements

`app/packaging/README.md` should include:

- install instructions
- environment variable setup
- simple `pied_piper.compress(...)` examples
- mixed input example
- note that images are passthrough and video returns an inline MP4 artifact

### 18.5 Installation flows

Local editable install:

```bash
python -m pip install --editable app/packaging
```

Build distributions:

```bash
cd app/packaging
python -m build
```

Publish later with Twine or trusted publishing, after confirming final project name availability.

## 19. Error Handling Rules

### 19.1 SDK-side configuration errors

Raise `ConfigurationError` when:

- `PIED_PIPER_BASE_URL` is missing and no explicit `base_url` is provided
- `PIED_PIPER_API_KEY` is missing and no explicit `api_key` is provided
- an unsupported input type is provided

### 19.2 SDK-side request errors

Raise `RequestError` or `ServerError` when:

- the HTTP request cannot be sent
- the response is malformed
- the server returns a non-401 error response that is request-level

### 19.3 Item-level failures

Do **not** throw away the whole response when one file fails extraction.

Preferred behavior:

- keep request success
- mark only the affected item as `failed`
- return `partial_success` overall

This is especially important for mixed batches.

## 20. Testing Plan

### 20.1 SDK tests

- normalize inline text input
- normalize `Path` input
- reject plain string file paths
- reject unsupported suffixes
- ensure multipart manifest generation preserves order
- ensure auth header is added
- ensure 401 maps to `AuthenticationError`

### 20.2 Backend unit tests

- auth dependency accepts valid bearer token
- auth dependency rejects missing/invalid bearer token
- multipart manifest parsing works
- text/image/video routing works
- phase 0 response statuses are correct

### 20.3 Backend integration tests

- text-only request
- mixed text + image request
- mixed text + video request
- bad document extraction returns item-level failure

## 21. Operational Notes

### 21.1 Local Modal development

Use:

```bash
modal serve app/inference/modal_app.py
```

This yields an ephemeral public URL and live reloads as files change.

### 21.2 Deployment

Use:

```bash
modal deploy app/inference/modal_app.py
```

The deployed function URL can then be surfaced in the SDK configuration as `PIED_PIPER_BASE_URL`.

### 21.3 Modal Secret creation

The hackathon secret should contain:

- `PIED_PIPER_API_KEY`

Recommended secret name:

- `pied-piper-backend`

## 22. Implementation Sequence

Recommended implementation order:

1. create `app/packaging` and make the SDK installable
2. create `app/inference` as a Modal FastAPI ASGI app
3. wire `GET /`, `GET /health`, and `POST /v1/compress`
4. implement auth dependency using the shared bearer token
5. implement SDK request building against the final contract
6. implement real text path
7. keep image passthrough stable
8. implement the real GPU-backed video clip selector without changing the SDK contract

## 23. Source Notes

The following decisions are directly grounded in current external documentation:

- Modal web apps:
  - use `@modal.asgi_app` when defining a full FastAPI app with multiple routes
- Modal scaling:
  - functions scale to zero by default when inactive
  - `min_containers`, `buffer_containers`, and `scaledown_window` control warm capacity
  - default idle window is 60 seconds
- Modal web endpoint limits:
  - request bodies can be up to 4 GiB
- Modal auth options:
  - Modal supports proxy auth tokens, but standard bearer auth in FastAPI is also supported
- Modal secrets:
  - attach a secret with `secrets=[modal.Secret.from_name(...)]` and read values from environment variables
- FastAPI multipart behavior:
  - files and forms can be combined
  - JSON body fields cannot be declared alongside multipart file/form fields in the same path operation
- HTTPX transport:
  - multipart uploads are first-class
  - files must be opened in binary mode
  - default inactivity timeout is 5 seconds
- Python packaging:
  - modern packaging should use `pyproject.toml`
  - `python -m pip install .`, `python -m pip install --editable .`, and `python -m build` should be used instead of invoking `setup.py` commands directly

## 24. External References

- Modal web endpoints guide: <https://modal.com/docs/guide/webhooks>
- Modal `asgi_app` reference: <https://modal.com/docs/reference/modal.asgi_app>
- Modal `fastapi_endpoint` reference: <https://modal.com/docs/reference/modal.fastapi_endpoint>
- Modal apps/functions guide: <https://modal.com/docs/guide/apps>
- Modal scaling guide: <https://modal.com/docs/guide/scale>
- Modal cold start guide: <https://modal.com/docs/guide/cold-start>
- Modal secrets guide: <https://modal.com/docs/guide/secrets>
- Modal `Secret` reference: <https://modal.com/docs/reference/modal.Secret>
- Modal pricing: <https://modal.com/pricing>
- Modal images guide: <https://modal.com/docs/guide/images>
- FastAPI request files docs: <https://fastapi.tiangolo.com/tutorial/request-files/>
- FastAPI request forms and files docs: <https://fastapi.tiangolo.com/tutorial/request-forms-and-files/>
- HTTPX quickstart: <https://www.python-httpx.org/quickstart/>
- HTTPX timeouts: <https://www.python-httpx.org/advanced/timeouts/>
- HTTPX exceptions: <https://www.python-httpx.org/exceptions/>
- Python Packaging User Guide, packaging projects: <https://packaging.python.org/en/latest/tutorials/packaging-projects/>
- Python Packaging User Guide, setup.py deprecation discussion: <https://packaging.python.org/en/latest/discussions/setup-py-deprecated/>
- Setuptools pyproject configuration: <https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html>
- Setuptools package discovery: <https://setuptools.pypa.io/en/stable/userguide/package_discovery.html>
- Setuptools data files/package data: <https://setuptools.pypa.io/en/stable/userguide/datafiles.html>
