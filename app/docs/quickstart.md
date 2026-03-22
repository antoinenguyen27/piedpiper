# Quickstart

This example does one thing end to end:

1. send a toy text string to Pied Piper
2. use the compressed output as context for an OpenAI API request
3. print the final response text

## 1. Install dependencies

```bash
python3 -m pip install piedpiper-sdk openai
```

## 2. Configure environment variables

```bash
export PIED_PIPER_API_KEY="replace-me"
export OPENAI_API_KEY="replace-me"
```

The SDK already points at the production Pied Piper service. Set `PIED_PIPER_BASE_URL` only if you need to override it for local development or a different deployment.

## 3. Run the example

Paste the example below into `quickstart.py`, then run:

```bash
python3 quickstart.py
```

## 4. What the script does

- compresses a toy block of text with `pied_piper.compress(...)`
- checks that the compression request succeeded
- sends the compressed text to the OpenAI Responses API
- prints `response.output_text`

Default fidelity behavior:

- if you do not pass `fidelity`, the SDK uses `0.9`
- higher `fidelity` preserves more content
- for text, `fidelity` maps to LLMLingua `rate`
- `fidelity=0.9` therefore means "keep roughly 90% of the original tokens", not "drop 90%"
- for video, `fidelity` maps directly to the target kept-duration budget before padding and merge
- the returned `compression_rate` is the observed output ratio `compressed_tokens / origin_tokens`, so it may differ from the requested `fidelity`

## 5. Example code

```python
from pathlib import Path

from openai import OpenAI
import pied_piper


TOY_TEXT = """
Pied Piper is acting as a preprocessing step before the LLM call.
The goal is to shorten verbose source material while keeping the main facts.
This toy example uses only inline text so the setup stays simple.
The compressed result is then forwarded into a normal OpenAI API request.
""".strip()


def main() -> None:
    compressed = pied_piper.compress(TOY_TEXT)

    if compressed.status == "failed" or not compressed.text:
        raise RuntimeError(f"Compression failed: status={compressed.status!r}")

    # Other supported file inputs:
    # compressed = pied_piper.compress(Path("notes.pdf"))
    # compressed = pied_piper.compress(Path("slides.pptx"))
    # compressed = pied_piper.compress(Path("draft.md"))

    # Multiple inputs in one request:
    # compressed = pied_piper.compress([Path("notes.pdf"), Path("slides.pptx")])
    # compressed = pied_piper.compress([TOY_TEXT, Path("notes.pdf"), Path("diagram.png")])
    # compressed = pied_piper.compress([TOY_TEXT, Path("notes.pdf"), Path("demo.mp4")])

    client = OpenAI()
    response = client.responses.create(
        model="gpt-5.4",
        input=(
            "Use the compressed context below to answer the question.\n\n"
            f"Compressed context:\n{compressed.text}\n\n"
            "Question: What is Pied Piper doing in this example?"
        ),
    )

    print("Compression status:", compressed.status)
    print("Compressed text:")
    print(compressed.text)
    print()
    print("OpenAI response:")
    print(response.output_text)


if __name__ == "__main__":
    main()
```

## 6. Notes

- `str` inputs are treated as inline text, not local file paths.
- Use `Path(...)` for files so the SDK uploads them correctly.
- The current OpenAI example uses the Responses API and the official Python SDK, following the current OpenAI quickstart:
  [Developer quickstart](https://developers.openai.com/api/docs/quickstart/#install-the-openai-sdk-and-run-an-api-call)
  and [Responses API reference](https://api.openai.com/v1/responses).
