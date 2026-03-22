# Pied Piper Python SDK

Pied Piper is a lightweight Python SDK for the Pied Piper multimodal compression service.

## Install

```bash
python -m pip install -e app/packaging
```

## Configure

Set your API key:

```bash
export PIED_PIPER_API_KEY="your-shared-api-key"
```

The SDK defaults to the production Pied Piper service URL. `PIED_PIPER_BASE_URL` is only needed if you want to override that for local development or a non-production deployment.

## Quickstart

```python
import pied_piper

result = pied_piper.compress("Long inline text to compress")
print(result.status)
print(result.text)
```

Default fidelity behavior:

- if you do not pass `fidelity`, the SDK uses `0.9`
- higher `fidelity` preserves more content
- for text, `fidelity` maps to LLMLingua `rate`
- `fidelity=0.9` therefore means "keep roughly 90% of the original tokens", not "drop 90%"
- for video, `fidelity` maps directly to the target kept-duration budget before padding and merge
- the returned `compression_rate` is the observed output ratio `compressed_tokens / origin_tokens`, so it may differ from the requested `fidelity`

## Mixed inputs

```python
from pathlib import Path

from pied_piper import Client

client = Client()
result = client.compress(
    [
        "inline text",
        Path("paper.pdf"),
        Path("diagram.png"),
        Path("demo.mp4"),
    ],
    fidelity=0.55,
)
```

Current behavior:

- text is extracted and compressed remotely
- images are accepted as passthrough items
- video inputs return an inline MP4 artifact on `item.output_file`

```python
video_item = next(item for item in result.items if item.modality == "video")
video_bytes = video_item.output_file.as_bytes()
```
