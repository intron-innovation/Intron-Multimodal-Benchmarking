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
├── qa_rubric.py            # 13-dimension Spoken QA clinical rubric
├── qa_rubric_judge.py      # LLM-as-judge scoring against that rubric
├── qa_rubric_evals.py      # dimension tables + judge-vs-human agreement
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

The **answering** half of Spoken QA: plays each spoken question to a model and
records its answer. Scoring happens elsewhere (see 2.1-2.3 and
[`docs/spoken_qa_pipeline.md`](../docs/spoken_qa_pipeline.md)).

#### Input

* `data/Spoken QA/meta_data.csv` (398 questions, 4 languages)

#### Output

* `results/spoken_qa/<model>_<language>.csv`

#### Models

Declared in the `QA_MODELS` registry - every wrapper shares the signature
`answer(input_audio, input_language, question, output_language) -> {"content": str}`:

| `--model`        | Provider model           | Output prefix            |
| ---------------- | ------------------------ | ------------------------ |
| `gemma4`         | Gemma 3n (local)         | `gemma4`                 |
| `gpt4o_audio_qa` | `gpt-4o-audio-preview`   | `gpt4o-audio-qa`         |
| `gemini_3_flash` | `gemini-3-flash-preview` | `gemini-3-flash-preview` |
| `qwen_qa`        | `qwen3.6-plus`           | `qwen-plus`              |

To add a model: write the wrapper in `models/`, add one `QA_MODELS` entry, add
its conda env to the QA block of `run_benchmarks.sh`.

#### Useful flags

* `--resume` - reuse answers already on disk; re-ask only missing or `"ERROR"`
  ones (a rate-limited run rarely finishes in one attempt)
* `--model-name` / `--output-prefix` - run a different provider revision into
  its own result files
* `--limit N` - smoke test

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

* Semantic similarity (COMET)

`evaluations.py` covers the automatic metrics only. The 13 clinical
dimensions Spoken QA is reported on come from the rubric scripts below.

---

### 2.1 `qa_rubric.py`

#### Purpose

Defines the **Spoken QA evaluation rubric**: the 13 dimensions, their anchors
and worked examples, the scoring prompt, and the few-shot example pool. This
is the same rubric the physician expert panel rates against.

Not run directly — imported by the two scripts below.

#### Dimensions

* 9 positive: `factuality`, `appropriatness`, `adequacy`, `expert_recall`,
  `identifies_uncertainty`, `empathy`, `clinical_reasoning`, `language_style`,
  `formatting_grammar`
* 4 negative: `hallucination`, `local_relevance`, `harm`,
  `poor_question_quality`

Scores stay in the **natural direction** of each rubric label (5 = most
harmful on `harm`), matching the expert-panel export, so judge and human
columns are directly comparable.

---

### 2.2 `qa_rubric_judge.py`

#### Purpose

Scores Spoken QA answers with an **LLM judge** on all 13 dimensions — the
scalable stand-in for a fresh expert-panel round.

#### Input

* Any CSV with `scenario`, `question`, `answer`
* Spoken QA = the `modality == "audio"` slice (`--modality audio`, the default)
* For benchmark predictions, point at the prediction column:
  `--answer-col hypothesis`

#### Output

* `results/spoken_qa_rubric/<judge>_scores.csv` — input columns preserved,
  13 score columns appended

#### Judges

| `--judge`  | Model                    | API key env         |
| ---------- | ------------------------ | ------------------- |
| `claude`   | claude-opus-4-7          | `ANTHROPIC_API_KEY` |
| `gpt`      | gpt-5.5                  | `OPENAI_API_KEY`    |
| `qwen`     | qwen3.6-plus (DashScope) | `DASHSCOPE_API_KEY` |
| `deepseek` | deepseek-reasoner        | `DEEPSEEK_API_KEY`  |

---

### 2.3 `qa_rubric_evals.py`

#### Purpose

Turns per-answer rubric scores into reportable tables.

#### Input

* `--human` expert-panel ratings (one row per answer per rater)
* `--judge-scores` optional judge scores for the same answers

#### Output (into `evaluations/spoken_qa/`)

* `spoken_qa_<dimension>.csv` — mean score per language x model, per dimension
* `spoken_qa_rubric_overall.csv` — mean across all 13 dimensions
* `rater_agreement.csv` — panel inter-rater reliability (Krippendorff's alpha)
* `judge_agreement_<judge>.csv`, `judge_overall_<judge>.csv`,
  `summary_<judge>.txt` — judge vs panel, per dimension and overall

#### Two things it handles that a naive join gets wrong

* **Answer identity.** `answer_id` names the question slot, not the answer —
  the same id recurs once per model that answered it. Answers are matched on
  `answer_id` + `model` + a hash of the answer text.
* **Score direction.** A judge CSV written on the "5 = best on every column"
  scale must be loaded with `--judge-scale best5`; comparing it to the panel
  unconverted flips the sign of the `harm` and `hallucination` correlations.

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
python scripts/qa_rubric_evals.py --human "data/Spoken QA/expert_panel_ratings.csv" --modality audio
```

The rubric step is skipped with a message when the expert-panel ratings are
not present locally.

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
