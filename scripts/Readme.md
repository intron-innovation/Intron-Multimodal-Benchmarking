# Scripts Module: Multilingual Speech Benchmark (Healthcare)

## Overview

The `scripts/` directory contains all **core logic for running benchmarks, evaluations, and report generation**.

This module orchestrates the full pipeline:

```text
data/ → scripts/ → results/ → evaluations/ → reports
```

It is designed to be:

* **Modular** (each task is independent)
* **Reproducible** (script-driven execution)
* **Extensible** (easy to plug in new models/tasks)

---

## Directory Structure

```bash id="r8m3kp"
scripts/
├── models/
├── qa_benchmark.py
├── transcription_benchmark.py
├── translation_benchmark.py
├── evaluations.py
├── report_gen.py
├── run_benchmarks.sh
├── run_evaluations.sh
├── setup.sh
```

---

## 1. Benchmark Scripts

These scripts generate **model predictions** and save them to `results/`.

---

### 1.1 `qa_benchmark.py`

#### Purpose

Runs **Spoken Question Answering (QA)** benchmarks.

#### Input

* `data/Spoken_QA/meta_data.csv`

#### Output

* `results/spoken_qa/<model>_<language>.csv`

#### Responsibilities

* Load QA dataset
* Run model inference
* Generate answers
* Save predictions

---

### 1.2 `transcription_benchmark.py`

#### Purpose

Runs **ASR (speech-to-text)** benchmarks.

#### Input

* `data/Transcription/meta_data.csv`

#### Output

* `results/transcription/<model>_<language>.csv`

#### Responsibilities

* Load audio + transcripts
* Run ASR model
* Generate predictions

---

### 1.3 `translation_benchmark.py`

#### Purpose

Runs **translation benchmarks**.

#### Input

* `data/Translation/meta_data.csv`

#### Output

* `results/translation/<model>_<language>.csv`

#### Responsibilities

* Load source text/audio
* Run translation model
* Save outputs

---

## 2. Evaluation Script

---

### `evaluations.py`

#### Purpose

Computes evaluation metrics for all tasks.

#### Input

* Model outputs from `results/`
* Ground truth from `data/`

#### Output

* Metrics stored in `evaluations/`

---

### Metrics Covered

#### Transcription

* WER (Word Error Rate)
* CER (Character Error Rate)

#### Translation

* BLEU
* chrF
* COMET

#### Spoken QA

* Semantic similarity (COMET-style)

---

## 3. Report Generation

---

### `report_gen.py`

#### Purpose

Generates **aggregated reports** from evaluation results.

#### Input

* `evaluations/`

#### Output

* Summary tables
* Comparative analysis

---

### Capabilities

* Per-language performance
* Cross-model comparison
* Task-level summaries

---

## 4. Shell Scripts

---

### 4.1 `setup.sh`

#### Purpose

Initial environment setup.

#### Responsibilities

* Install dependencies
* Configure environment

---

### 4.2 `run_benchmarks.sh`

#### Purpose

Runs all benchmark tasks sequentially.

#### Equivalent to:

```bash id="7k2dmp"
python scripts/transcription_benchmark.py
python scripts/translation_benchmark.py
python scripts/qa_benchmark.py
```

---

### 4.3 `run_evaluations.sh`

#### Purpose

Runs evaluation pipeline.

#### Equivalent to:

```bash id="p8m2qs"
python scripts/evaluations.py
```

---

## 5. `models/` Directory

### Description

Contains model-specific logic and wrappers.

### Responsibilities

* Load models
* Handle inference
* Abstract model differences

---

## Workflow

### Step 1: Setup Environment

```bash id="t4c9hz"
bash scripts/setup.sh
```

---

### Step 2: Run Benchmarks

```bash id="6y8vjq"
bash scripts/run_benchmarks.sh
```

---

### Step 3: Run Evaluations

```bash id="h9w2lp"
bash scripts/run_evaluations.sh
```

---



---

## Design Principles

### 1. Separation of Concerns

* Benchmarking ≠ Evaluation ≠ Reporting
* Each script has a single responsibility

---

### 2. Reproducibility

* All experiments are script-driven
* No manual steps required

---

### 3. Extensibility

To add a new model:

1. Add implementation in `models/`
2. Integrate into benchmark script
3. Run pipeline

---

### 4. Data Consistency

All scripts rely on:

* `meta_data.csv` files
* Consistent IDs and paths

---

## Adding a New Model

### Step 1: Implement Model Wrapper

Add logic in:

```bash id="p7m4yx"
scripts/models/
```

---

### Step 2: Integrate into Benchmark Script

Update:

* `qa_benchmark.py`
* `transcription_benchmark.py`
* `translation_benchmark.py`

---

### Step 3: Run Pipeline

```bash id="z5q8tn"
bash scripts/run_benchmarks.sh
```

---




## Summary

The `scripts/` module is the **engine of the benchmark**:

* Runs models
* Computes metrics
* Generates reports

It ensures:

* Reproducibility
* Scalability
* Clean separation of pipeline stages

---
