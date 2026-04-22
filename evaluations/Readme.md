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
│   └── spoken_qa_comet.csv
├── transcriptions/
├── translations/
│   ├── translation_bleu.csv
│   ├── translation_chrf.csv
│   └── translation_comet.csv
```

---

## 1. Spoken QA Evaluation (`spoken_qa/`)

### File

* `spoken_qa_comet.csv`

### Description

Evaluates **semantic correctness of answers** using COMET-style metrics.

Unlike exact-match metrics, COMET captures:

* Meaning similarity
* Contextual correctness
* Fluency

---

### Expected Columns (Typical)

| Column       | Description            |
| ------------ | ---------------------- |
| `id`         | Sample identifier      |
| `language`   | Language of sample     |
| `prediction` | Model-generated answer |
| `reference`  | Ground truth answer    |
| `score`      | COMET score            |
| `model`      | Model name             |

---

### Notes

* Spoken QA evaluation focuses on **semantic equivalence**, not exact string match
* Particularly important in multilingual settings

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
