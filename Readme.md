# Multilingual Speech Benchmark for Healthcare

## Overview

This repository contains a **comprehensive benchmarking framework for evaluating speech models in healthcare settings**, with a focus on:

* **Automatic Speech Recognition (ASR)** → Transcription
* **Speech Translation** → Cross-lingual understanding
* **Spoken Question Answering (Spoken QA)** → End-to-end reasoning from speech

The project targets **multilingual, low-resource, and African languages**, evaluating how well modern speech and multimodal models perform in **real-world healthcare scenarios**.

---

## Motivation

Speech technologies are increasingly being deployed in healthcare for:

* Clinical documentation
* Patient interaction
* Decision support systems

However, most systems are:

* Optimized for **high-resource languages**
* Not robust to **accent, dialect, or domain-specific terminology**

This project aims to:

* Provide a **standardized benchmark**
* Evaluate models across **languages and tasks**
* Identify **performance gaps in healthcare contexts**

---

## Repository Structure

```
.
├── data/
├── docs/
├── evaluations/
├── notebooks/
├── outputs/
├── pairwise_agreement_analysis/
├── requirements/
├── results/
├── scripts/
├── .env
├── .gitignore
└── readme.md
```

---

## Directory Breakdown

### `data/`

Contains all datasets used for benchmarking:

* `audio/` → Raw speech data
* `Spoken_QA/` → Spoken question-answering datasets
* `Transcription/` → Ground truth transcripts
* `Translation/` → Parallel text for translation

Each subdirectory includes a **`meta_data.csv`** describing:

* File paths
* Language
* Task-specific annotations

📌 **Note:**
This directory has its own detailed `README.md` explaining:

* Data format
* Annotation schema
* Preprocessing steps

---

### `evaluations/`

Contains evaluation outputs and metrics:

* `spoken_qa/` → QA evaluation results (e.g., COMET)
* `transcriptions/` → ASR evaluation results
* `translations/` → Translation metrics:

  * BLEU
  * chrF
  * COMET

Each evaluation file is stored as `.csv` for easy analysis.

📌 A dedicated `README.md` explains:

* Metric definitions
* Evaluation pipeline

---

### `results/`

Model outputs for each task:

#### Spoken QA

* `gemma4_English.csv`
* `gemma4_Hausa.csv`
* `gemma4_Pidgin.csv`
* `gemma4_Yoruba.csv`

#### Translation

Includes outputs for multiple languages:

* Afrikaans, Amharic, Hausa, Igbo, Kinyarwanda, Swahili, Yoruba, Zulu, etc.

These files represent **raw model predictions** before evaluation.

---

### `scripts/`

Core benchmarking and evaluation logic:

* `qa_benchmark.py` → Spoken QA evaluation pipeline
* `transcription_benchmark.py` → ASR benchmarking
* `translation_benchmark.py` → Translation benchmarking
* `evaluations.py` → Metric computation
* `report_gen.py` → Aggregated report generation

Shell scripts:

* `run_benchmarks.sh` → Run all benchmarks
* `run_evaluations.sh` → Compute evaluation metrics
* `setup.sh` → Environment setup

---

### `requirements/`

Dependency management:

* `requirements_gemma4.txt`
* `requirements_medasr.txt`

These correspond to different model environments.

---

### `pairwise_agreement_analysis/`

Analysis tools for:

* Model agreement
* Cross-model consistency
* Error correlation

---






---

### `outputs/`

Intermediate outputs and logs generated during experiments.

---

## Supported Tasks

### 1. Transcription (ASR)

* Input: Audio
* Output: Text
* Metrics: WER, CER

---

### 2. Translation

* Input: Speech/Text
* Output: Target language text
* Metrics:

  * BLEU
  * chrF
  * COMET

---

### 3. Spoken Question Answering

* Input: Audio question + context
* Output: Answer
* Metrics:

  * COMET (semantic quality)

---

## Supported Languages

The benchmark includes multiple languages (17 for transcription, 16 for translation and 4 for spoken question and answering), with emphasis on African and low-resource settings:


---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repo_url>
cd <repo_name>
```

---

### 2. Set Up Environment

```bash
bash scripts/setup.sh
```

Or manually:

```bash
pip install -r requirements/requirements_gemma4.txt
```

---

### 3. Run Benchmarks

#### Transcription

```bash
bash scripts/run_benchmark.sh transcription
```

#### Translation

```bash
bash scripts/run_benchmark.sh translation
```

#### Spoken QA

```bash
bash scripts/run_benchmark.sh qa
```
#### All Task

```bash
bash scripts/run_benchmark.sh all
```
---

### 4. Run Evaluations

```bash
bash scripts/run_evaluations.sh
```

---

### 5. Generate Reports

```bash
python scripts/report_gen.py
```

---

## Workflow Summary

1. Prepare data (`data/`)
2. Run model inference → outputs saved in `results/`
3. Run evaluation → metrics saved in `evaluations/`
4. Analyze results via notebooks or reports

---

## Design Principles

* **Modular** → Each task is independent
* **Reproducible** → Script-based pipelines
* **Extensible** → Easy to add new models/languages
* **Research + Production Ready**

---

## Notes on Documentation

Each major directory contains its own **detailed `README.md`**:

* `data/README.md` → Data schema and format
* `evaluations/README.md` → Metrics and evaluation logic
* `scripts/README.md` → Benchmark pipeline details
* `results/README.md` → Output format

Please refer to these for deeper details.

---

## Next Steps

* Add more **Models**
* Integrate **Metrics like time of each model**
* Include **human evaluation benchmarks for spoken qa**

---

## Contribution Guidelines

* Follow existing folder structure
* Add documentation for any new module
* Ensure reproducibility of results
* Include evaluation outputs

---

## License

(To be specified)

---

## Contact

For questions, collaboration, or research inquiries, please reach out to the maintainers. research[at]intron[dot]io

---
