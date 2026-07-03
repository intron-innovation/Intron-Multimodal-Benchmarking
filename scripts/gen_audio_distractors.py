"""
Generate AUDIO distractor answers (plausible-but-wrong) for a random 15% of clips
per subset, feeding the WAV to gemini-flash. Output: results/audio_qa/distractors_<subset>.csv
(flows through combine as model='distractors', modality=audio).
"""
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q
import qa_audio_benchmark as A

OUT_DIR = os.path.join(A.REPO_ROOT, "results", "audio_qa")
MODEL_ID = "gemini-3.1-pro-preview"
FRAC = 0.15
SEED = 42

PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a concise answer in "
    "English which should seem right by casual readers or inexperienced medical personnel "
    "but should be wrong by medical standards. The question is spoken in the provided "
    "audio, in {language} language, spoken in Nigeria. Listen to the audio and answer the "
    "question. Provide the answer ONLY in English language. No other language should be "
    "present in your answer. Be concise, presenting your answer in 5 sentences or less."
)

NATIVE_PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a concise answer which "
    "should seem right by casual readers or inexperienced medical personnel but should be "
    "wrong by medical standards. The question is spoken in the provided audio, in {language} "
    "language, spoken in Nigeria. Listen to the audio and answer the question in {language} "
    "language, the same language the question is asked in. Be concise, presenting your "
    "answer in 5 sentences or less."
)


def gen(args):
    path, prompt = args
    for _ in range(3):
        out = A.google_audio(MODEL_ID, path, prompt, q.get_key("google"))
        if not str(out).startswith("ERROR"):
            return out
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=str, default="")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--native", action="store_true", help="answer in the question's own language")
    args = ap.parse_args()
    want = {x.strip() for x in args.subsets.split(",") if x.strip()}
    out_dir = args.out_dir
    template = NATIVE_PROMPT if args.native else PROMPT

    df = A.load_audio_index()  # audio_id, language, subset, local_path, reference, url
    picks = []
    for subset, g in df.groupby("subset"):
        if want and subset not in want:
            continue
        n = math.ceil(len(g) * FRAC)
        picks.append(g.sample(n=n, random_state=SEED))
        print(f"{subset}: {len(g)} clips -> {n} distractors")
    sample = pd.concat(picks, ignore_index=True)

    work = [(p, template.format(language=str(l).capitalize()))
            for p, l in zip(sample["local_path"], sample["language"])]
    with ThreadPoolExecutor(max_workers=4) as ex:
        answers = list(ex.map(gen, work))
    sample["model_answer"] = answers
    sample["requested_model"] = "Distractor (gemini-3.1-pro audio)"
    sample["model_id"] = MODEL_ID

    cols = ["audio_id", "language", "subset", "reference", "url",
            "model_answer", "requested_model", "model_id"]
    cols = [c for c in cols if c in sample.columns]
    os.makedirs(out_dir, exist_ok=True)
    for subset, gs in sample.groupby("subset"):
        gs[cols].to_csv(os.path.join(out_dir, f"distractors_{subset}.csv"), index=False)
        print(f"wrote distractors_{subset}.csv ({len(gs)} rows)")
    errs = sample["model_answer"].astype(str).str.startswith("ERROR").sum()
    print(f"total {len(sample)} audio distractors, {errs} errors")


if __name__ == "__main__":
    main()
