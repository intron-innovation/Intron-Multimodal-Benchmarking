# Results Module: Multilingual Speech Benchmark (Healthcare)

## Overview

The `results/` directory contains **raw model outputs** generated during benchmarking.

These outputs represent predictions from different models across tasks:

* **Transcription (ASR)**
* **Translation**
* **Spoken Question Answering (Spoken QA)**

This module acts as the **bridge between inference and evaluation**:

* Inputs → `data/`
* Predictions → `results/`
* Metrics → `evaluations/`

---

## Directory Structure

```bash id="2kq9lm"
results/
├── spoken_qa/
│   ├── gemma4_English.csv
│   ├── gemma4_Hausa.csv
│   ├── gemma4_Pidgin.csv
│   └── gemma4_Yoruba.csv
├── spoken_qa_rubric/            # per-answer 13-dimension judge scores
│   ├── <judge>_scores.csv       # not committed - see samples/
│   └── samples/
│       └── claude_scores_sample.csv
├── transcription/
├── translation/
│   ├── gemma4_afrikaans.csv
│   ├── gemma4_akan.csv
│   ├── gemma4_amharic.csv
│   ├── gemma4_arabic.csv
│   ├── gemma4_french.csv
│   ├── gemma4_hausa.csv
│   ├── gemma4_igbo.csv
│   ├── gemma4_kinyarwanda.csv
│   ├── gemma4_pedi.csv
│   ├── gemma4_sesotho.csv
│   ├── gemma4_shona.csv
│   ├── gemma4_swahili.csv
│   ├── gemma4_tswane.csv
│   ├── gemma4_xhosa.csv
│   ├── gemma4_yoruba.csv
│   └── gemma4_zulu.csv
```

---

## Spoken QA answers (`spoken_qa/`)

Produced by `scripts/qa_benchmark.py` - one file per model per language, with
the model's answer in `hypothesis` and the reference answer in `reference`.

A failed provider call is recorded as `"ERROR"` in `hypothesis` rather than
dropped, so coverage stays visible. As of the last run, `qwen-plus` has 63
such failures out of 398; `python scripts/qa_benchmark.py --model qwen_qa
--resume` re-asks only those. Scoring treats `"ERROR"` as the model's answer,
so clear them before quoting that model's metrics.

---

## Spoken QA rubric scores (`spoken_qa_rubric/`)

Per-answer scores on the 13-dimension clinical rubric, produced by
`scripts/qa_rubric_judge.py`:

```bash
python scripts/qa_rubric_judge.py --csv "data/Spoken QA/expert_panel_ratings.csv" \
    --judge claude --modality audio
```

Each file preserves every input column verbatim and appends the 13 score
columns, stored in the natural direction of each rubric label (5 = most
harmful on `harm`) so they line up with the expert-panel ratings.
`scripts/qa_rubric_evals.py` aggregates them into `evaluations/spoken_qa/`.

These files are **not committed** — they carry the full clinical exchange.
`samples/claude_scores_sample.csv` holds five records showing the format.

---

## Naming Convention

All result files follow the format:

```bash id="b5x3nt"
<model_name>_<language>.csv
```

### Examples

* `gemma4_hausa.csv`
* `gemma4_yoruba.csv`
* `gemma4_afrikaans.csv`

---

## 1. Spoken QA Results (`spoken_qa/`)

### Description

Contains model outputs for **spoken question answering tasks**.

### Expected Columns

| Column       | Description            |
| ------------ | ---------------------- |
| `id`         | Sample identifier      |
| `question`   | Input question         |
| `prediction` | Model-generated answer |
| `reference`  | Ground truth answer    |
| `language`   | Language code          |
| `model`      | Model name             |

---

### Notes

* Predictions should reflect **final answers**, not intermediate reasoning
* Must align with entries in `data/Spoken_QA/meta_data.csv`

---

## 2. Transcription Results (`transcription/`)

### Description

Contains **ASR outputs** from models.

### Expected Columns

| Column       | Description             |
| ------------ | ----------------------- |
| `id`         | Sample identifier       |
| `audio_path` | Input audio             |
| `prediction` | Transcribed text        |
| `reference`  | Ground truth transcript |
| `language`   | Language code           |
| `model`      | Model name              |

---

### Notes

* Ensure text normalization consistency with ground truth
* Outputs directly feed into WER/CER evaluation

---

## 3. Translation Results (`translation/`)

### Description

Contains model outputs for **translation tasks**.

### Expected Columns

| Column        | Description              |
| ------------- | ------------------------ |
| `id`          | Sample identifier        |
| `source_text` | Input text               |
| `prediction`  | Translated output        |
| `reference`   | Ground truth translation |
| `source_lang` | Source language          |
| `target_lang` | Target language          |
| `model`       | Model name               |

---

### Notes

* Must align with `data/Translation/meta_data.csv`
* Used for BLEU, chrF, and COMET evaluation

---

## Workflow Integration

### Step 1: Generate Results

Run benchmark scripts:

```bash id="z6p0m2"
bash run_benchmark.sh
```

---

### Step 2: Save Outputs

All predictions must be saved in:

```bash id="q2d1cv"
results/
```

---

### Step 3: Evaluate

Run:

```bash id="m9v4jk"
bash scripts/run_evaluations.sh
```

---

## Design Principles

### 1. Raw Outputs Only

* No post-processing beyond minimal formatting
* Evaluation scripts handle scoring

---

### 2. Consistency with Data

* Each row must map to a sample in `data/`
* IDs must be preserved

---

### 3. Model-Agnostic

* Supports multiple models
* Easy comparison across systems

---

### 4. Language Separation

* Each language stored independently
* Enables per-language benchmarking

---





## Notes

* Files should be UTF-8 encoded
* Avoid modifying results after evaluation (for reproducibility)
* Keep raw outputs for auditability

---


## Summary

The `results/` module stores **all model predictions**, serving as the foundation for:

* Evaluation
* Analysis
* Reporting

Well-structured results ensure:

* Reliable benchmarking
* Fair comparison across models
* Reproducible research

---
