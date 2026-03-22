# spec.md — Video Context Compression (Pre-Inference Clip Selector)

## 1. Overview

This system takes a video (MP4) as input and returns a **shortened MP4** by **removing low-value clips** while preserving:

* temporal continuity
* semantic coverage
* clip integrity

The system is **model-agnostic downstream** and operates **before any LLM/VLM inference**.

---

## 2. Core Principles

### 2.1 Easy Negatives, Not Hard Positives

The system removes:

* redundant clips
* low-novelty segments
* repetitive visual content

It does **not attempt deep semantic understanding**.

---

### 2.2 Clip-Level Compression Only

* Clips are **kept or dropped as whole units**
* No frame-level deletion inside kept clips
* No temporal distortion

---

### 2.3 Conservative by Default

* Preserve broad coverage
* Avoid catastrophic information loss
* Compression is a **budgeted reduction**, not aggressive minimization

---

### 2.4 Task-Agnostic First

Default mode does **not require text prompts**
Optional text-conditioning is supported but not required

---

## 3. Inputs

### Required

* `video.mp4`

### Optional

* `prompt: str` (text conditioning)
* `fidelity: float` (0.0–1.0, higher preserves more; maps directly to the target kept-duration budget)
* `mode: {conservative, balanced, aggressive}` (optional preset override; mutually exclusive with `fidelity`)

---

## 4. Outputs

* `compressed.mp4`
* `metadata.json`

### metadata.json

```json
{
  "original_duration": 760.2,
  "output_duration": 442.5,
  "reduction_ratio": 0.417,
  "clips_total": 84,
  "clips_kept": 53,
  "clips_removed": 31
}
```

Notes:

* output duration is approximate rather than exact
* the selector targets `original_duration * fidelity` before padding and merge
* use either direct numeric `fidelity` or explicit `mode`, but not both

---

## 5. Pipeline

## 5.1 Video Decode

Extract:

* duration
* fps
* frame access
* timestamps

---

## 5.2 Shot Segmentation

### Method

* **TransNet V2**

### Output

List of clips:

```python
clip = {
  "start": float,
  "end": float,
  "duration": float
}
```

---

## 5.3 Clip Cleanup

### Merge rules

* merge clips < **1.0s** into neighbors
* remove pathological fragmentation
* ensure minimum temporal coherence

---

## 5.4 Frame Sampling per Clip

| Clip Duration | Frames |
| ------------- | ------ |
| < 3s          | 1      |
| 3–10s         | 3      |
| > 10s         | 5      |

### Sampling strategy

Even spacing with boundary margin (~10%):

Example (3 frames):

* 20%, 50%, 80%

---

## 5.5 Embedding Model

### Model

* CLIP (ViT-B/32 class)

### Processing

* flatten all frames
* batch encode on GPU
* regroup by clip

---

## 5.6 Clip Embedding Construction

### Task-Agnostic Mode

```python
clip_embedding = mean(frame_embeddings)
```

Normalize after pooling.

---

### Text-Conditioned Mode

```python
text_embedding = encode_text(prompt)
```

Per-frame similarity:

```python
sim = cosine(frame_emb, text_emb)
```

Clip score:

```python
clip_score = mean(top_2(similarities))
```

---

## 6. Selection Policies

---

# 6.1 Task-Agnostic (Default)

## Objective

Maximize:

* diversity
* coverage
* non-redundancy

Minimize:

* redundancy

---

## Algorithm: Greedy Novelty + Temporal Coverage

### Initialize

* keep first clip
* last kept = first

### For each clip in order:

Compute:

```python
similarity = cosine(clip_emb, last_kept_emb)
```

Keep if:

```python
similarity < novelty_threshold
OR
time_since_last_kept > max_gap
```

---

### Recommended defaults

```python
novelty_threshold = 0.90–0.95
max_gap = 20–40 seconds
```

---

## Budget Constraint

Stop keeping clips when:

```python
total_kept_duration >= target_keep_ratio * total_duration
```

---

# 6.2 Text-Conditioned Mode

## Score

```python
score = α * relevance + β * novelty + γ * coverage
```

Where:

* relevance = top-2 similarity
* novelty = 1 - similarity to recent kept clips

### Defaults

```python
α = 0.65
β = 0.25
γ = 0.10
```

---

## Selection

Greedy selection under duration budget:

* prioritize higher score
* enforce temporal spread
* avoid local clustering

---

## 7. Compression Budget

### Modes

| Mode         | Keep Ratio |
| ------------ | ---------- |
| Conservative | 0.7–0.8    |
| Balanced     | 0.5–0.7    |
| Aggressive   | 0.3–0.5    |

---

### Behavior

* treated as **target**, not hard constraint
* allow ±5–10% deviation

---

## 8. Output Construction

### Rules

* use original clips unchanged
* concatenate in original order

---

### Enhancements

#### Merge small gaps

If gap < 0.5–1.0s → merge clips

#### Padding

Add:

* 0.2–0.5s before and after clips

---

## 9. Performance Characteristics

### Expected behavior on T4

* 1–5 frames per clip
* batched CLIP inference
* total latency: seconds to tens of seconds

---

## 10. Failure Modes (and Mitigations)

| Issue              | Mitigation                   |
| ------------------ | ---------------------------- |
| Overcompression    | conservative defaults        |
| Missing events     | temporal coverage constraint |
| Repetition remains | novelty filtering            |
| Jittery output     | no intra-clip modification   |
| Overfitting prompt | novelty + coverage terms     |

---

## 11. Non-Goals

This system does **not**:

* perform full video understanding
* detect all important events
* optimize exact token usage
* replace downstream LLM/VLM

---

## 12. Summary

### Core pipeline

```text
MP4
 → TransNet segmentation
 → clip cleanup
 → 1/3/5 frame sampling
 → CLIP embeddings
 → clip scoring
 → greedy selection (budgeted)
 → stitch original clips
 → MP4
```

---

## 13. Key Insight

> **Do not compress moments. Compress the set of moments.**

---
