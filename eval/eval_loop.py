import os
import re
import json
import time
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import openai
import anthropic
import google.generativeai as genai
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. Configuration & Constants
# ==========================================
COMPRESSION_API_URL = "http://your-compression-service.internal/compress"
USE_COMPRESSION = True  # Toggle for Baseline (False) vs. Test (True)

# Set these in your environment or replace here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key")


# ==========================================
# 2. Data Structures (From your pipeline)
# ==========================================
@dataclass(slots=True)
class SourceText:
    item_id: str
    index: int
    source_name: str
    text: str


@dataclass(slots=True)
class TextOptions:
    chunk_chars: int = 4000
    overlap_chars: int = 300
    fidelity: Optional[float] = 0.5
    target_token: Optional[int] = None
    drop_consecutive: bool = False


# ==========================================
# 3. Compression API Wrapper
# ==========================================
def call_compression_api(text: str, fidelity: float = 0.5) -> str:
    """
    Calls your hosted LLMLingua2 compression service.
    """
    if not USE_COMPRESSION:
        return text

    payload = {
        "text": text,
        "fidelity": fidelity,
        "use_token_level_filter": True,
        "force_tokens": ["\n", "A)", "B)", "C)", "D)", "E)", "F)", "G)", "H)", "I)", "J)"]
    }

    try:
        response = requests.post(COMPRESSION_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("compressed_text", text)
    except Exception as e:
        print(f"Compression API Error: {e}")
        return text


# ==========================================
# 4. Frontier Model SDK Wrapper
# ==========================================
def call_frontier_model(prompt: str, provider: str = "openai") -> str:
    """
    Handles calls to OpenAI, Claude, or Gemini.
    """
    system_prompt = "You are an expert. Answer the following multiple-choice question by thinking step-by-step and then stating the correct letter."

    if provider == "openai":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return resp.choices[0].message.content

    elif provider == "claude":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.content[0].text

    elif provider == "gemini":
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        resp = model.generate_content(f"{system_prompt}\n\n{prompt}")
        return resp.text

    return ""


# ==========================================
# 5. Eval Helpers (Formatting & Parsing)
# ==========================================
def format_mmlu_prompt(item: Dict[str, Any]) -> str:
    """Formats MMLU-Pro entry into a structured MCQ prompt."""
    options = item['options']
    formatted_options = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
    return (
        f"Question: {item['question']}\n"
        f"Options:\n{formatted_options}\n"
        f"Answer: Let's think step by step."
    )


def extract_answer_letter(response_text: str) -> Optional[str]:
    """Uses regex to find the A-J answer in the model's output."""
    patterns = [
        r"answer is \(?([A-J])\)?",
        r"Answer:\s*([A-J])",
        r"\b([A-J])\b(?=\s*is the correct)"
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: check last 10 chars for a single letter
    last_chars = response_text.strip()[-10:]
    match = re.search(r"([A-J])", last_chars, re.IGNORECASE)
    return match.group(1).upper() if match else None


# ==========================================
# 6. Main Evaluation Loop
# ==========================================
def run_evaluation(provider: str = "openai", num_samples: int = 50, fidelity: float = 0.5):
    print(f"--- Starting Eval: {provider.upper()} | Compression: {USE_COMPRESSION} (Fidelity: {fidelity}) ---")

    # Load MMLU-Pro (using subset for speed)
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test_piedpiper", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    stats = {"correct": 0, "total": 0, "errors": 0}
    results_log = []

    for item in tqdm(dataset, desc="Evaluating"):
        raw_prompt = format_mmlu_prompt(item)

        # Step 1: Compress
        final_prompt = call_compression_api(raw_prompt, fidelity=fidelity)

        # Step 2: Inference
        try:
            full_response = call_frontier_model(final_prompt, provider=provider)
            predicted = extract_answer_letter(full_response)
            actual = item['answer']

            is_correct = (predicted == actual)
            if is_correct:
                stats["correct"] += 1
            stats["total"] += 1

            results_log.append({
                "id": item.get("question_id"),
                "actual": actual,
                "predicted": predicted,
                "correct": is_correct,
                "compressed": USE_COMPRESSION
            })

            # Rate limiting safety for Tier 1 APIs
            time.sleep(0.5)

        except Exception as e:
            print(f"\nError processing item: {e}")
            stats["errors"] += 1

    # Final Summary
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print("\n" + "=" * 30)
    print(f"EVALUATION COMPLETE")
    print(f"Provider:   {provider}")
    print(f"Accuracy:   {acc:.2%}")
    print(f"Correct:    {stats['correct']}/{stats['total']}")
    print(f"Errors:     {stats['errors']}")
    print("=" * 30)

    # Save results to disk
    with open(f"results_{provider}_{'compressed' if USE_COMPRESSION else 'raw'}.json", "w") as f:
        json.dump(results_log, f, indent=2)


if __name__ == "__main__":
    # Example usage:
    # 1. Run baseline (USE_COMPRESSION = False)
    # 2. Run test_piedpiper (USE_COMPRESSION = True)
    run_evaluation(provider="openai", num_samples=10, fidelity=0.4)