# Quickstart

This example does one thing end to end:

1. send a toy text string to Pied Piper
2. use the compressed output as context for an OpenAI API request
3. print the final response text

The runnable script lives at [docs/examples/piedpiper_openai_quickstart.py](/Users/an/Documents/piedpiper/docs/examples/piedpiper_openai_quickstart.py).

## 1. Install dependencies

From the repo root:

```bash
python3 -m pip install -e app/packaging
python3 -m pip install openai
```

## 2. Configure environment variables

```bash
export PIED_PIPER_BASE_URL="https://<your-modal-url>.modal.run"
export PIED_PIPER_API_KEY="replace-me"
export OPENAI_API_KEY="replace-me"
```

## 3. Run the example

```bash
python3 docs/examples/piedpiper_openai_quickstart.py
```

## 4. What the script does

- compresses a toy block of text with `pied_piper.compress(...)`
- checks that the compression request succeeded
- sends the compressed text to the OpenAI Responses API
- prints `response.output_text`

## 5. Example code

```python
from pathlib import Path

from openai import OpenAI
import pied_piper


toy_text = """
Pied Piper is acting as a preprocessing step before the LLM call.
The goal is to shorten verbose source material while keeping the main facts.
This toy example uses only inline text so the setup stays simple.
"""

compressed = pied_piper.compress(toy_text)

# Other supported file inputs:
# compressed = pied_piper.compress(Path("notes.pdf"))
# compressed = pied_piper.compress(Path("slides.pptx"))
# compressed = pied_piper.compress(Path("draft.md"))

# Multiple inputs in one request:
# compressed = pied_piper.compress([Path("notes.pdf"), Path("slides.pptx")])
# compressed = pied_piper.compress([toy_text, Path("notes.pdf"), Path("diagram.png")])
# compressed = pied_piper.compress([toy_text, Path("notes.pdf"), Path("demo.mp4")])

client = OpenAI()
response = client.responses.create(
    model="gpt-5.4",
    input=(
        "Use the compressed context below to answer the question.\n\n"
        f"Compressed context:\n{compressed.text}\n\n"
        "Question: What is Pied Piper doing in this example?"
    ),
)

print(response.output_text)
```

## 6. Notes

- `str` inputs are treated as inline text, not local file paths.
- Use `Path(...)` for files so the SDK uploads them correctly.
- The current OpenAI example uses the Responses API and the official Python SDK, following the current OpenAI quickstart:
  [Developer quickstart](https://developers.openai.com/api/docs/quickstart/#install-the-openai-sdk-and-run-an-api-call)
  and [Responses API reference](https://api.openai.com/v1/responses).
