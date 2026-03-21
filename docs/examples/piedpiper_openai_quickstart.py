from __future__ import annotations

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
