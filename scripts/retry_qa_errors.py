"""
Retry only the rows whose model_answer starts with 'ERROR' in results/text_qa and
results/audio_qa, re-invoking the model with a couple of backoff attempts. Updates
the CSVs in place. Safe to run repeatedly; files still being written (e.g. an
in-progress model) simply have no ERROR rows yet or are skipped by --skip-labels.
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as T
import qa_audio_benchmark as A


def retry_answer(kind, m, row):
    for attempt in range(3):
        if kind == "text":
            out = T.answer_one(m, T.PROMPT.format(
                language=str(row["language"]).capitalize(), question=str(row["question"])))
        else:
            out = A.answer_one(m, row["local_path"],
                               A.AUDIO_PROMPT.format(language=str(row["language"]).capitalize()))
        if not str(out).startswith("ERROR"):
            return out
        time.sleep(5 * (attempt + 1))
    return out


def run(kind, d, registry, skip):
    reg = {m["label"]: m for m in registry}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        label = os.path.basename(f).split("_", 1)[0]
        if label in skip or label not in reg:
            continue
        df = pd.read_csv(f)
        mask = df["model_answer"].astype(str).str.startswith("ERROR")
        if not mask.any():
            continue
        for i in df.index[mask]:
            ans = retry_answer(kind, reg[label], df.loc[i])
            df.at[i, "model_answer"] = ans
            print(f"  {os.path.basename(f)} row {i}: {str(ans)[:70]}")
        df.to_csv(f, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-labels", type=str, default="", help="comma list of labels to skip")
    args = ap.parse_args()
    skip = {x.strip() for x in args.skip_labels.split(",") if x.strip()}
    print("== retry TEXT ==")
    run("text", os.path.join(T.REPO_ROOT, "results", "text_qa"), T.REGISTRY, skip)
    print("== retry AUDIO ==")
    run("audio", os.path.join(A.REPO_ROOT, "results", "audio_qa"), A.REGISTRY, skip)
    print("done")


if __name__ == "__main__":
    main()
