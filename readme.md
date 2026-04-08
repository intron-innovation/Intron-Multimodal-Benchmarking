# 🚀 Speech & Language Evaluation Benchmark

This repository benchmarks state-of-the-art models across:
- 🗣️ Transcription (ASR)
- 🌍 Translation
- ❓ Question Answering (QA)

The focus is on **low-resource and African language settings**, with domain-specific evaluation (e.g., medical).

---

## 📊 Results

Results are automatically generated from CSV files for reproducibility.

👉 See full results here:  
➡️ [RESULTS.md](./RESULTS.md)

---

## 📁 Project Structure

A detailed breakdown of the repository structure is available here:

➡️ [docs/STRUCTURE.md](./docs/STRUCTURE.md)

---

## ⚙️ How to Update Results

1. Update the CSV files:
   - `data/transcription_results.csv`
   - `data/translation_results.csv`
   - `data/qa_results.csv`

2. Regenerate results:
   ```bash
   python scripts/generate_results_md.py

## 📊 Results

<!-- RESULTS_START -->

# 📊 Results

## 🗣️ Transcription (Gabby → Abdul)

### 📌 Description
This task evaluates ASR performance.

### 📏 Evaluation Metric: WER / MWER

### 📈 Results

| Model | WER | MWER |
|-------|-----|------|
| Intron Sahara | nan | nan |
| Meta-Omni-ASR-7B-LLM/CTC | nan | nan |
| OpenAI 4o Transcribe | nan | nan |
| Qwen-3.5-Omni | nan | nan |
| Qwen-3-Omni (offline) | nan | nan |
| Google Medical STT | nan | nan |
| AWS Transcribe Medical | nan | nan |
| Azure Speech | nan | nan |
| Gemma-3 MedASR | nan | nan |
| Gemma-4-E4B | nan | nan |


## 🌍 Translation (Busayo → Abdul)

### 📌 Description
Translation quality evaluation.

### 📏 Evaluation Metric: AfriCOMET

### 📈 Results

| Model | AfriCOMET |
|-------|-----------|
| Gemini-3-Flash | nan |
| GPT-4o Audio | nan |
| Qwen-3.5-Omni | nan |
| Azure Speech | nan |
| Google Translate | nan |
| Gemma-4-E4B | nan |


## ❓ Question Answering (Aka → Abdul)

### 📌 Description
QA evaluation.

### 📏 Evaluation Metric: Accuracy (LLM-as-a-Judge)

### 📈 Results

| Model | Accuracy |
|-------|----------|
| Gemini-3-Flash | nan |
| GPT-audio-1.5 | nan |
| Qwen-3.5-Omni | nan |
| Gemma-4-E4B | nan |


<!-- RESULTS_END -->