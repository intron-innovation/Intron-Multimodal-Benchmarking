# Data Module: Multilingual Speech Benchmark (Healthcare)

## Overview

The `data/` directory contains all datasets used for benchmarking speech models across three core healthcare tasks:

* **Transcription (ASR)**
* **Translation**
* **Spoken Question Answering (Spoken QA)**

Each task is defined through a **`meta_data.csv`** file that serves as the **single source of truth** for data loading and evaluation.

---

## Directory Structure

```bash
data/
├── audio/
├── Spoken_QA/
│   └── meta_data.csv
├── Transcription/
│   └── meta_data.csv
├── Translation/
│   └── meta_data.csv
```

---

## 1. `audio/`

### Description

Contains all raw audio files referenced across tasks.

### Requirements

* Format: `.wav` (recommended)
* Sampling rate: 16kHz
* Mono channel preferred

---

## 2. Spoken QA (`Spoken_QA/meta_data.csv`)

### Description

This is the **most complex dataset** in the project, supporting:

* Spoken QA
* Multimodal inputs (audio, image, video, PDF)
* Annotator metadata
* Scenario-based reasoning

---

### Key Column Groups

#### **Core Identifiers**

* `audio_id`, `sentence_id`, `doc_id`, `seq_id`
* `question_id`, `answer_id`
* `project_id`, `project_name`

---

#### **Audio Information**

* `audio_path`
* `audio_duration`
* `audio_recording_path`
* `audio_upload_path`
* `audio_path_transcribed`
* `audio_transcribed_duration`

---

#### **Text Fields**

* `prompt`
* `question`
* `answer`
* `new_text`
* `feedback_text`
* `scenario`
* `scenario_question`

---

#### **Language & Translation**

* `language`
* `src_lang`, `src_text`
* `target_lang`, `target_text`
* `transcribed_audio_language`

---

#### **Quality & Annotation**

* `quality`
* `difficulty`
* `category`
* `prompt_scenario_similarity`
* `match_seq_id`

---

#### **Annotator Metadata**

* `user_id`, `first_name`, `last_name`, `email`
* `level`, `status`
* `age_group`, `gender`, `accent`
* `discipline`, `education`, `institution`
* `clinical_experience`

---

#### **Geographic Information**

* `city`, `country`

---

#### **Feedback Metrics**

* `num_pos_feedback_received`
* `num_neg_feedback_received`
* `num_feedback_received`
* `neg_percent`

---

#### **Multimodal Inputs**

* `image_upload_path`
* `video_upload_path`
* `pdf_upload_path`

---

#### **Data Paths (Processed)**

* `data_audio_recording_path`
* `data_audio_path`
* `data_image_path`
* `data_video_path`
* `data_audio_path_transcribed`

---

### Notes

* This dataset supports **end-to-end spoken reasoning tasks**
* Some columns may be **optional depending on scenario**
* Empty fields should be handled gracefully in pipelines

---

## 3. Transcription (`Transcription/meta_data.csv`)

### Description

Dataset for **automatic speech recognition (ASR)**.

### Columns

| Column       | Description                |
| ------------ | -------------------------- |
| `audio_path` | Path to audio file         |
| `duration`   | Audio length (seconds)     |
| `text`       | Ground truth transcription |
| `language`   | Language code              |
| `source`     | Dataset source             |
| `root`       | Root directory for audio   |

---

### Example

```csv
audio_path,duration,text,language,source,root
ha_001.wav,4.2,majiyyaci yana da zazzabi,ha,clinical,/data/audio
```

---

## 4. Translation (`Translation/meta_data.csv`)

### Description

Dataset for **speech/text translation**.

### Columns

| Column          | Description            |
| --------------- | ---------------------- |
| `id`            | Sample identifier      |
| `transcription` | Source text            |
| `translation`   | Target text            |
| `language`      | Language code (source) |
| `speaker_id`    | Speaker identifier     |
| `gender`        | Speaker gender         |
| `duration`      | Audio duration         |
| `source`        | Dataset source         |
| `audio_path`    | Path to audio          |
| `audio_root`    | Root directory         |

---

### Example

```csv
id,transcription,translation,language,speaker_id,gender,duration,source,audio_path,audio_root
1,majiyyaci yana da zazzabi,the patient has fever,ha,spk1,male,4.2,clinical,ha_001.wav,/data/audio
```

---

## Data Design Principles

### 1. Metadata-Driven

All datasets are controlled via `meta_data.csv` files.
No hardcoded paths should be used in scripts.

---

### 2. Task Separation

* Spoken QA → reasoning + multimodal
* Transcription → ASR
* Translation → cross-lingual

Each dataset is independent but can share audio.

---

### 3. Healthcare Focus

All samples are expected to reflect:

* Clinical conversations
* Symptoms and diagnosis
* Patient-provider interactions

---

### 4. Multilingual Support

Languages are explicitly labeled and must be consistent across:

* Data
* Results
* Evaluations

---

## Adding New Data

### Step 1: Add Audio

Place audio in:

```bash
data/audio/
```

---

### Step 2: Update Metadata

Update the appropriate `meta_data.csv` file.

---

### Step 3: Validate

Ensure:

* No missing required columns
* Correct file paths
* Consistent language codes

---

## Common Pitfalls

* Missing or incorrect `audio_path`
* Inconsistent language labeling
* Misaligned transcription vs audio
* Overloaded Spoken QA entries with unused fields

---

## Notes

* CSV encoding must be **UTF-8**
* Large datasets should use external storage if needed
* Keep column names unchanged (used in scripts)

---

## Summary

The `data/` module is the backbone of the benchmark:

* Structured metadata
* Multitask support
* Rich annotations for research and production

Proper formatting ensures:

* Reliable evaluation
* Reproducibility
* Scalability

---
