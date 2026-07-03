"""
Generate distractor answers (deliberately plausible-but-wrong) for a random 15% of
the unique questions per language, using Claude. A validity check: a good judge
should score these low. Output: results/text_qa/distractor_<subset>.csv (flows
through combine as model='distractor').
"""
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q

REPO = q.REPO_ROOT
IN_DIR = os.path.join(REPO, "outputs", "gates_transcription")
OUT_DIR = os.path.join(REPO, "results", "text_qa")
MODEL_ID = "gemini-3.1-pro-preview"
FRAC = 0.15
SEED = 42

PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a concise answer in "
    "English which should seem right by casual readers or inexperienced medical personnel "
    "but should be wrong by medical standards. The question will be in {language} language, "
    "spoken in Nigeria. Provide the answer ONLY in English language. No other language "
    "should be present in your answer. Be concise, presenting your answer in 5 sentences "
    "or less. Here is the question: {question}"
)

NATIVE_PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a concise answer which "
    "should seem right by casual readers or inexperienced medical personnel but should be "
    "wrong by medical standards. The question will be in {language} language, spoken in "
    "Nigeria. Provide the answer in {language} language, the same language the question is "
    "asked in. Be concise, presenting your answer in 5 sentences or less. Here is the "
    "question: {question}"
)


def call_model(prompt):
    for attempt in range(3):
        try:
            out = q.call_google(MODEL_ID, prompt, q.get_key("google"))
        except Exception as e:  # noqa: BLE001
            out = f"ERROR: {type(e).__name__}: {str(e)[:150]}"
        if not str(out).startswith("ERROR"):
            return (out or "").strip()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=str, default="", help="comma list to (re)generate; default all")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--native", action="store_true", help="answer in the question's own language")
    args = ap.parse_args()
    want = {x.strip() for x in args.subsets.split(",") if x.strip()}
    out_dir = args.out_dir
    template = NATIVE_PROMPT if args.native else PROMPT

    frames = []
    for f in sorted(__import__("glob").glob(os.path.join(IN_DIR, "sahara_*.csv"))):
        d = pd.read_csv(f)
        d["subset"] = os.path.basename(f)[len("sahara_"):-4]
        frames.append(d.rename(columns={"hypothesis": "question"}))
    df = pd.concat(frames, ignore_index=True)

    # 15% random sample per subset (each subset = one recording set)
    picks = []
    for subset, g in df.groupby("subset"):
        if want and subset not in want:
            continue
        n = math.ceil(len(g) * FRAC)
        picks.append(g.sample(n=n, random_state=SEED))
        print(f"{subset}: {len(g)} rows -> {n} distractors")
    sample = pd.concat(picks, ignore_index=True)

    prompts = [template.format(language=str(l).capitalize(), question=str(qs))
               for l, qs in zip(sample["language"], sample["question"])]
    with ThreadPoolExecutor(max_workers=8) as ex:
        answers = list(ex.map(call_model, prompts))
    sample["model_answer"] = answers
    sample["requested_model"] = "Distractor (gemini-3.1-pro)"
    sample["model_id"] = MODEL_ID

    cols = ["audio_id", "language", "subset", "question", "reference", "url",
            "model_answer", "requested_model", "model_id"]
    cols = [c for c in cols if c in sample.columns]
    os.makedirs(out_dir, exist_ok=True)
    for subset, gs in sample.groupby("subset"):
        gs[cols].to_csv(os.path.join(out_dir, f"distractors_{subset}.csv"), index=False)
        print(f"wrote distractors_{subset}.csv ({len(gs)} rows)")
    errs = sample["model_answer"].astype(str).str.startswith("ERROR").sum()
    print(f"total {len(sample)} distractors, {errs} errors")


if __name__ == "__main__":
    main()
