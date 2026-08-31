# Evaluations Module: Multilingual Speech Benchmark (Healthcare)

## Overview

The `evaluations/` directory contains **quantitative evaluation results** for all benchmarked tasks:

* **Transcription (ASR)**
* **Translation**
* **Spoken Question Answering (Spoken QA)**

This module is responsible for:

* Storing **computed metrics**
* Enabling **cross-model comparisons**
* Supporting **analysis and reporting**

All evaluation outputs are stored as **`.csv` files** for transparency and reproducibility.

---

## Directory Structure

```bash id="x4v2kp"
evaluations/
├── spoken_qa/
│   ├── spoken_qa_comet.csv              # semantic similarity to the reference answer
│   ├── spoken_qa_factuality.csv         # one file per rubric dimension (13 total)
│   ├── spoken_qa_appropriatness.csv
│   ├── ...
│   ├── spoken_qa_rubric_overall.csv     # mean across all 13 dimensions
│   ├── rater_agreement.csv              # expert-panel inter-rater reliability
│   ├── judge_agreement_<judge>.csv      # LLM judge vs expert panel, per dimension
│   ├── judge_overall_<judge>.csv
│   └── summary_<judge>.txt
├── transcriptions/
├── translations/
│   ├── translation_bleu.csv
│   ├── translation_chrf.csv
│   └── translation_comet.csv
```

---

## 1. Spoken QA Evaluation (`spoken_qa/`)

> Answering and scoring are separate halves of the Spoken QA pipeline; both
> are documented in [`docs/spoken_qa_pipeline.md`](../docs/spoken_qa_pipeline.md).
> This file covers the scoring half.

Spoken QA is evaluated on **two levels**, and they answer different questions.
A COMET score alone cannot tell a locally infeasible answer from a feasible
one, or a dangerous answer from a safe one — which is what a clinical
benchmark has to report — so the semantic metric is paired with a clinical
rubric.

---

### 1.1 Semantic similarity — COMET

#### File

* `spoken_qa_comet.csv`

#### Description

COMET between the model's answer and the reference answer, per language and
model. Computed by `scripts/evaluations.py::spoken_qa_evals`.

Captures meaning similarity, contextual correctness, and fluency. It does
**not** capture factuality, safety, local feasibility, or completeness.

#### Format

Rows are languages, columns are models — the same layout as the WER and BLEU
tables.

| Column     | Description                       |
| ---------- | --------------------------------- |
| `Language` | Language of the evaluated subset  |
| `<model>`  | System-level COMET for that model |

---

### 1.2 Clinical rubric — 13 dimensions

The dimension scores the benchmark reports come from a **13-point Likert
rubric** for AI-generated clinical responses aimed at Community Health
Extension Workers (CHEWs) in rural Nigerian primary care. Every Spoken QA
answer (the `modality == "audio"` slice) is rated on:

**Positive dimensions** (5 = best)

| Dimension                | Measures                                                  |
| ------------------------ | --------------------------------------------------------- |
| `factuality`             | Correctness against scientific consensus                   |
| `appropriatness`         | Fit to a resource-limited PHC / CHEW scope of practice     |
| `adequacy`               | Whether every sub-question asked is resolved               |
| `expert_recall`          | Red flags and pearls clinicians commonly omit              |
| `identifies_uncertainty` | Flags missing data and asks targeted clarifying questions  |
| `empathy`                | Patient-centred content, counselling guidance              |
| `clinical_reasoning`     | Differential thinking, justified management                |
| `language_style`         | Register and audience fit for a community health setting   |
| `formatting_grammar`     | Surface mechanics: spelling, punctuation, formatting       |

**Negative dimensions** (5 = most of the bad thing)

| Dimension               | Measures                                                    |
| ----------------------- | ----------------------------------------------------------- |
| `hallucination`         | Fabricated drugs, doses, references; repetitive padding      |
| `local_relevance`       | Recommends locally unavailable or inappropriate management   |
| `harm`                  | Advice that could plausibly cause harm                       |
| `poor_question_quality` | Rates the question, not the answer                           |

The rubric text — the anchors and worked examples for each dimension — lives
in `scripts/qa_rubric.py`.

#### Who does the rating

* **Physician expert panel** — the ground truth. Ratings arrive as one row
  per (answer, rater); `scripts/qa_rubric_evals.py` averages across raters
  and reports the panel's own agreement in `rater_agreement.csv`
  (ordinal Krippendorff's alpha).
* **LLM judge** — `scripts/qa_rubric_judge.py` replicates the same rubric so
  new models can be scored without a new panel round. A judge's scores are
  only usable once `judge_agreement_<judge>.csv` shows it tracks the panel.

#### Output files

| File                               | Contents                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| `spoken_qa_<dimension>.csv`        | Mean score per language x model, one file per dimension       |
| `spoken_qa_rubric_overall.csv`     | Mean of all 13 dimensions (negatives flipped so higher = better) |
| `rater_agreement.csv`              | Krippendorff's alpha per dimension across human raters        |
| `judge_agreement_<judge>.csv`      | Pearson/Spearman/MAE/RMSE/bias per dimension, judge vs panel  |
| `judge_overall_<judge>.csv`        | Single-row overall agreement for that judge                   |
| `summary_<judge>.txt`              | Human-readable version of the above                           |

#### Scale convention — read before comparing anything

Scores are stored in the **natural direction of each rubric label**: 5 = most
harmful on `harm`, 5 = most fabricated on `hallucination`. That is how the
rater UI presents them (physicians move a slider labelled "Answer could cause
harm") and how the panel export stores them, so judge and human columns can
be compared directly.

The negative dimensions are flipped to good-direction in exactly one place:
the `spoken_qa_rubric_overall` table, where averaging across all 13 would
otherwise mix two opposing scales.

If a judge CSV was written on the "5 = best on every column" scale, convert
it on load with `--judge-scale best5`. Comparing the two scales without
converting turns genuine agreement on `harm` and `hallucination` into a
*negative* correlation.

#### Running it

```bash
# 1. Score answers with an LLM judge (needs the provider's API key)
python scripts/qa_rubric_judge.py \
    --csv "data/Spoken QA/expert_panel_ratings.csv" \
    --judge claude --modality audio

# 2. Dimension tables + panel reliability, and judge agreement if supplied
python scripts/qa_rubric_evals.py \
    --human "data/Spoken QA/expert_panel_ratings.csv" \
    --judge-scores results/spoken_qa_rubric/claude_scores.csv \
    --judge claude --modality audio
```

`bash scripts/run_evaluations.sh` runs step 2 automatically when the panel
ratings are present.

#### Data availability

The expert-panel ratings and the per-answer judge scores are **not committed**
— they contain clinical content contributed by identified health workers.
Five-record samples showing the exact schema are in:

* `data/Spoken QA/samples/expert_panel_ratings_sample.csv`
* `results/spoken_qa_rubric/samples/claude_scores_sample.csv`

---

## 2. Transcription Evaluation (`transcriptions/`)

### Description

Evaluates **ASR performance**.

### Metrics

* **WER (Word Error Rate)**
* **CER (Character Error Rate)**

---

### Expected Output Format

| Column       | Description             |
| ------------ | ----------------------- |
| `id`         | Sample identifier       |
| `language`   | Language                |
| `prediction` | Model transcript        |
| `reference`  | Ground truth transcript |
| `wer`        | Word Error Rate         |
| `cer`        | Character Error Rate    |
| `model`      | Model name              |

---

### Notes

* Lower WER/CER indicates better performance
* Important for **clinical accuracy** in healthcare

---

## 3. Translation Evaluation (`translations/`)

### Files

* `translation_bleu.csv`
* `translation_chrf.csv`
* `translation_comet.csv`

---

### Metrics

#### **BLEU**

* Measures n-gram overlap
* Good for surface-level similarity

#### **chrF**

* Character-level F-score
* More robust for morphologically rich languages

#### **COMET**

* Neural metric for semantic similarity
* Best for multilingual evaluation

---

### Expected Columns

| Column        | Description       |
| ------------- | ----------------- |
| `id`          | Sample identifier |
| `source_lang` | Source language   |
| `target_lang` | Target language   |
| `prediction`  | Model output      |
| `reference`   | Ground truth      |
| `score`       | Metric score      |
| `model`       | Model name        |

---

### Notes

* Use COMET as primary metric for research conclusions
* BLEU/chrF provide complementary insights

---

## Evaluation Pipeline

### Step 1: Generate Model Outputs

Outputs are stored in:

```bash id="3c4c6w"
results/
```

---

### Step 2: Run Evaluation

```bash id="9y7j7h"
bash scripts/run_evaluations.sh
```

or individually:

```bash id="d6z6pq"
python scripts/evaluations.py
```

---

### Step 3: Store Results

All computed metrics are saved in:

```bash id="xk3kqk"
evaluations/
```

---

## Design Principles

### 1. Metric Separation

Each metric is stored in its own file:

* Easier comparison
* Cleaner analysis

---

### 2. Reproducibility

* All evaluations are script-based
* No manual computation

---

### 3. Model-Agnostic

* Supports multiple models (e.g., Gemma, Whisper, etc.)
* Results are comparable across models

---

### 4. Language-Aware

* Metrics are computed per language
* Enables multilingual benchmarking

---

## Adding New Evaluations

### Step 1: Add Model Outputs

Place predictions in:

```bash id="5j5f2x"
results/
```

---

### Step 2: Update Evaluation Script

Modify:

```bash id="s1yq4s"
scripts/evaluations.py
```

---

### Step 3: Run Evaluation

```bash id="c3k4hs"
bash scripts/run_evaluations.sh
```

---



## Summary

The `evaluations/` module provides:

* Standardized metrics
* Reproducible evaluation
* Cross-lingual performance insights

It is critical for:

* Model comparison
* Research reporting
* Identifying weaknesses in healthcare speech systems

---
