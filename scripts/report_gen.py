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


def inject_into_readme(results_md):
    print("🔍 Reading readme.md...")
    with open("readme.md", "r") as f:
        readme = f.read()
        print(readme[:500])  # Print first 500 chars to verify content

    start_tag = "<!-- RESULTS_START -->"
    end_tag = "<!-- RESULTS_END -->"

    if start_tag in readme and end_tag in readme:
        before = readme.split(start_tag)[0]
        after = readme.split(end_tag)[1]

        new_readme = (
            before
            + start_tag + "\n\n"
            + results_md + "\n\n"
            + end_tag
            + after
        )

        with open("readme.md", "w") as f:
            f.write(new_readme)

        print("✅ readme.md updated with results!")
    else:
        print("⚠️ Tags not found in readme.md")


# Update main()
def main():
    transcription_df = pd.read_csv("outputs/transcription_results.csv")
    translation_df = pd.read_csv("outputs/translation_results.csv")
    qa_df = pd.read_csv("outputs/qa_results.csv")

    md = "# 📊 Results\n\n"

    # (same generation code as before...)
    md += "## 🗣️ Transcription (Gabby → Abdul)\n\n"
    md += "### 📌 Description\nThis task evaluates ASR performance.\n\n"
    md += "### 📏 Evaluation Metric: WER / MWER\n\n"
    md += generate_transcription_table(transcription_df) + "\n\n"

    md += "## 🌍 Translation (Busayo → Abdul)\n\n"
    md += "### 📌 Description\nTranslation quality evaluation.\n\n"
    md += "### 📏 Evaluation Metric: AfriCOMET\n\n"
    md += generate_translation_table(translation_df) + "\n\n"

    md += "## ❓ Question Answering (Aka → Abdul)\n\n"
    md += "### 📌 Description\nQA evaluation.\n\n"
    md += "### 📏 Evaluation Metric: Accuracy (LLM-as-a-Judge)\n\n"
    md += generate_qa_table(qa_df)

    # Save standalone file
    with open("docs/RESULTS.md", "w") as f:
        f.write(md)

    # 🔥 Inject into README
    inject_into_readme(md)

    print("✅ RESULTS.md + readme.md updated!")

if __name__ == "__main__":
    main()