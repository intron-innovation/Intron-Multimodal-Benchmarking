import pandas as pd

def generate_transcription_table(df):
    md = "### 📈 Results\n\n"
    md += "| Model | WER | MWER |\n"
    md += "|-------|-----|------|\n"
    for _, row in df.iterrows():
        md += f"| {row['model']} | {row['wer']} | {row['mwer']} |\n"
    return md


def generate_translation_table(df):
    md = "### 📈 Results\n\n"
    md += "| Model | AfriCOMET |\n"
    md += "|-------|-----------|\n"
    for _, row in df.iterrows():
        md += f"| {row['model']} | {row['africomet']} |\n"
    return md


def generate_qa_table(df):
    md = "### 📈 Results\n\n"
    md += "| Model | Accuracy |\n"
    md += "|-------|----------|\n"
    for _, row in df.iterrows():
        md += f"| {row['model']} | {row['accuracy']} |\n"
    return md


def main():
    transcription_df = pd.read_csv("outputs/transcription_results.csv")
    translation_df = pd.read_csv("outputs/translation_results.csv")
    qa_df = pd.read_csv("outputs/qa_results.csv")

    md = "# 📊 Results\n\n"

    # Transcription Section
    md += "## 🗣️ Transcription\n\n"
    md += "### 📌 Description\n"
    md += "This task evaluates automatic speech recognition (ASR) performance on domain-specific speech.\n\n"
    md += "### 📏 Evaluation Metric: WER / MWER\n"
    md += "- WER: Measures transcription errors.\n"
    md += "- MWER: Average WER across samples.\n\n"
    md += generate_transcription_table(transcription_df)
    md += "\n\n"

    # Translation Section
    md += "## 🌍 Translation n\n"
    md += "### 📌 Description\n"
    md += "Evaluates translation quality for low-resource and African languages.\n\n"
    md += "### 📏 Evaluation Metric: AfriCOMET\n"
    md += "- Measures semantic similarity and fluency.\n\n"
    md += generate_translation_table(translation_df)
    md += "\n\n"

    # QA Section
    md += "## ❓ Question Answering \n\n"
    md += "### 📌 Description\n"
    md += "Evaluates correctness of answers generated from speech-derived content.\n\n"
    md += "### 📏 Evaluation Metric: Accuracy (LLM-as-a-Judge)\n"
    md += "- Uses Claude 4.6 Opus as judge.\n"
    md += "- Validated via correlation with human evaluation (≥ 0.8).\n\n"
    md += generate_qa_table(qa_df)

    with open("RESULTS.md", "w") as f:
        f.write(md)

    print("✅ RESULTS.md generated successfully!")


if __name__ == "__main__":
    main()