![Pied Piper banner](./assets/banner.png)

# Pied Piper

Pied Piper is a Python SDK for compressing text and file inputs before sending them into an LLM workflow.

Full documentation is available [here](https://piedpiper-pi.vercel.app/).

[Find out more on our website](https://piedpiper-2dzt.vercel.app/).

## Access

You need a Pied Piper API key before you can use the SDK. Request one from [Antoine](https://www.linkedin.com/in/antoine-n-633b84200/).

## Install

```bash
python3 -m pip install piedpiper-sdk openai
```

## Quickstart

Set your credentials:

```bash
export PIED_PIPER_API_KEY="replace-me"
export OPENAI_API_KEY="replace-me"
```

The SDK already points at the production Pied Piper service. Set `PIED_PIPER_BASE_URL` only if you need to override it for local development or another deployment.

Run this example:

```python
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

Run it with:

```bash
python3 quickstart.py
```

## Notes

- `pied_piper.compress(...)` accepts inline text directly.
- Use `Path(...)` for local files so the SDK uploads them correctly.
- If you do not pass `fidelity`, the SDK uses `0.9`.
