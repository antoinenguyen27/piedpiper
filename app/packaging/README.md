# Pied Piper Python SDK

Pied Piper is a lightweight Python SDK for the Pied Piper multimodal compression service.

## Install

```bash
python -m pip install -e app/packaging
```

## Configure

Set these environment variables:

```bash
export PIED_PIPER_BASE_URL="https://your-modal-url"
export PIED_PIPER_API_KEY="your-shared-api-key"
```

## Quickstart

```python
import pied_piper

result = pied_piper.compress("Long inline text to compress", fidelity=0.33)
print(result.status)
print(result.text)
```

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
