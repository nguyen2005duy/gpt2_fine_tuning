# Fine-Tuning Vietnamese GPT-2 for Mathematical Word Problems

> **Group 17** · UET – VNU  
> Vũ Huy Công · Nguyễn Mạnh Cường · Nguyễn Đức Duy

---

## Overview

This project fine-tunes [`NlpHUST/gpt2-vietnamese`](https://huggingface.co/NlpHUST/gpt2-vietnamese) — a GPT-2-based causal language model — to solve Vietnamese mathematical word problems (MWPs). It runs entirely **offline** on Kaggle within a **3-hour GPU budget**.

The system combines two core ideas:
1. **Two-stage curriculum learning** via LoRA — train on simple arithmetic first, then complex reasoning.
2. **Ensemble decoding** — generate 20 stochastic samples per question, remove outliers with IQR, and aggregate answers by majority vote.

**Final validation score: 1,801 / 10,000 (18.01%)**

---

## Team Contributions

| Member | Role | Share |
|---|---|---|
| Nguyễn Mạnh Cường | Inference pipeline | 33% |
| Nguyễn Đức Duy | Data preprocessing | 33% |
| Vũ Huy Công | Training pipeline | 33% |

---

## Results Summary

| Configuration | Raw Score | Score₁₀ |
|---|---|---|
| Baseline (greedy, no fine-tuning) | 344 | 0.344 |
| Single-stage LoRA + greedy | 802 | 0.802 |
| LoRA + Curriculum + greedy | 831 | 0.831 |
| LoRA + Curriculum (Freeze) + Ensemble | 1,782 | 1.782 |
| **Final (Freeze + Ensemble + Post-correct)** | **1,801** | **1.801** |

---

## Data Processing

### Dataset Format

Each record in `train.json` / `valid.json` contains:

```python
{
  "original_question_vi": "...",   # original Vietnamese question
  "query_vi":             "...",   # (possibly augmented) question
  "response_vi":          "...",   # solution + answer
  "type":                 "..."    # e.g. GSM_Rephrased, MATH_SV, ...
}
```

### Filtering

| Step | Description |
|---|---|
| Language filter | Dropped 272 English records by checking alphanumeric balance vs. Vietnamese characters |
| Semantic filter | Removed unsolvable problems containing phrases like *"không đủ thông tin"* |
| Complexity filter | Removed problems with LaTeX tokens like `\log`, `\sqrt`, `\sin`, polynomials, matrices, etc. (`COMPLEX_PATTERNS`) |
| Structural filter | Dropped records missing an explicit answer token or with unconvertible `\frac` after cleaning |
| Token length | Kept only records ≤ 372 tokens (GPT-2 Vietnamese tokenizer), retaining ~95% of data |

### Normalization

- Standardized answer marker: `Câu trả lời là` / `\boxed{}` → `Đáp án là`
- Removed inline `$` signs and converted `\cdot` → `*`
- Translated LaTeX symbols to plain Vietnamese text (e.g. `\triangle` → *tam giác*, `\deg` → *độ*)
- Removed duplicate answer strings

### Curriculum Datasets

| Stage | File | Size | Content |
|---|---|---|---|
| Stage 1 | `stage1_augmented_arithmetic.csv` | ~35,000 records | Basic arithmetic (add/subtract/multiply/divide), numerically augmented |
| Stage 2 | `stage2_augmented_gsm_only.csv` | ~48,000 records | Multi-step GSM reasoning; MATH types excluded; cross-augmented AnsAug ↔ Rephrased |

**Stage 1 augmentation:** Numbers scaled by factors `[2, 3, 4, 5, 6, 7, 8, 10, 1.5, 2.5]` plus random additive offsets (≤ 50). Token budget for Stage 2: 8,300,000 tokens.

### Prompt Template

```
Câu hỏi: {question}
Lời giải: 
```

---

## Training

### Method: LoRA (Low-Rank Adaptation)

Applied to four projection layers per Transformer block: `c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`. `fan_in_fan_out=True` is required for GPT-2's transposed Conv1D weights.

### Two-Stage Curriculum

**Stage 1 — Arithmetic Foundation**

| Parameter | Value |
|---|---|
| LoRA rank / alpha | r=16, α=32 |
| Epochs | 2 |
| Learning rate | 5×10⁻³ |
| Batch size (per device) | 16 |
| Gradient accumulation | 2 (effective batch = 32) |
| Warmup ratio | 0.03 |

**Stage 2 — Complex Reasoning**

After Stage 1, the adapter is merged permanently into the base weights (`merge_and_unload()`), then a fresh higher-rank adapter is trained on Stage 2 data.

| Parameter | Value |
|---|---|
| LoRA rank / alpha | r=128, α=256 |
| Epochs | 2 |
| Learning rate | 5×10⁻³ |
| Batch size (per device) | 16 |
| Gradient accumulation | 1 (effective batch = 16) |
| Warmup ratio | 0.0 |

**Why Freeze instead of Continued?**  
Merging Stage 1 before Stage 2 prevents the higher-rank adapter from interfering with the arithmetic gradient signal learned in Stage 1. Preliminary experiments confirmed this approach yields higher validation scores.

### Shared Settings (Both Stages)

| Setting | Value |
|---|---|
| LR scheduler | Cosine annealing |
| Weight decay | 0.01 |
| Mixed precision | FP16 (GPU) |
| Max sequence length | 380 tokens |
| Random seed | 42 |
| `pad_token_id` = `eos_token_id` | 50256 |

### Arithmetic Post-Correction

A rule-based module scans generated text for expressions of the form `A op B = C`, evaluates the left-hand side, and replaces an incorrect right-hand side. Applied immediately after `model.generate()` before decoding.

---

## Inference

### Ensemble Decoding (Self-Consistency)

Generates **K = 20** stochastic samples per question, then aggregates answers.

| Parameter | Value |
|---|---|
| Samples (K) | 20 |
| Temperature | 0.7 |
| Top-k | 40 |
| Top-p (nucleus) | 0.90 |
| Max new tokens | 256 |
| Batch size | 4 (left-padded) |

### Answer Extraction

Searches generated text for the anchor *"Đáp án là"*. Extracted strings are parsed by `parse_number()`, which handles Vietnamese decimal notation (comma separator), thousands-separator commas, LaTeX fractions/roots, and safe `eval()` via a restricted namespace (`sqrt`, `pi` only).

### IQR Outlier Removal

Before aggregation, outliers are removed using the interquartile range method (fence = 1.5×IQR). Skipped if fewer than 4 numeric values are available or IQR = 0.

### Three-Tier Aggregation

1. **Majority Vote** — if one answer has a strictly highest count, return it.
2. **Closest-to-Mean** — on a tie, return the sample closest to the mean of cleaned answers.
3. **Mean Override** — inject the arithmetic mean into the solution text if no majority exists.

**Strategy distribution on validation set:**

| Strategy | Count | Total Score | Avg. Score |
|---|---|---|---|
| vote | 723 | 1,554 | 2.149 |
| closest_to_mean | 276 | 228 | 0.826 |
| no_valid_answer | 1 | 0 | 0.000 |

---

## Evaluation Metric

Relative error per example:

```
ε = |ŷ − y| / max(1, |y|)
```

| Condition | Score |
|---|---|
| ε ≤ 0.01 | 10 |
| ε ≤ 0.10 | 5 |
| ε ≤ 0.50 | 1 |
| Unanswerable / inaccurate | 0 |

Final metric: `score₁₀ = raw_score / N`

---

## Computational Budget

| Stage | Duration |
|---|---|
| Data preprocessing | ~10 min |
| Training (2-stage) | ~2 hr |
| Inference (K=20, full validation) | ~30 min |
| **Total** | **~2.8 hr** |

---

## References

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*, arXiv 2021
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Bengio et al., *Curriculum Learning*, ICML 2009
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, ICLR 2023
- *Intermediate Fine-Tuning Improves Mathematical Reasoning in Smaller Models*, OpenReview 2024
