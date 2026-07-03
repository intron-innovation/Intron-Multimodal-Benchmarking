"""
Generate distractor answers for the human-eval set: COUNT = ceil(0.15 * MODELS * base),
where MODELS defaults to 11 (the intended full model count) and base = rows in the subset.
Questions are sampled WITH REPLACEMENT to reach COUNT (subsets have < COUNT unique questions).
gemini-3.1-pro; English prompt unless --native. Writes audio_id/language/subset/question/
reference/url/model_answer so build_human_eval can consume it.
"""
import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q
import gen_distractors as gd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True, help="sahara transcription csv for the subset")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--models", type=int, default=11)
    ap.add_argument("--n-asr", type=int, default=2, help="number of ASR sources (multiplier)")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = pd.read_csv(args.in_file).rename(columns={"hypothesis": "question"})
    base = len(d)
    count = math.ceil(args.frac * args.models * base * args.n_asr)
    samp = d.sample(n=count, replace=True, random_state=args.seed).reset_index(drop=True)
    samp["subset"] = args.subset

    tmpl = gd.NATIVE_PROMPT if args.native else gd.PROMPT
    prompts = [tmpl.format(language=str(l).capitalize(), question=str(qs))
               for l, qs in zip(samp["language"], samp["question"])]
    with ThreadPoolExecutor(max_workers=8) as ex:
        answers = list(ex.map(gd.call_model, prompts))
    samp["model_answer"] = answers
    samp["requested_model"] = "Distractor (gemini-3.1-pro)"
    samp["model_id"] = gd.MODEL_ID

    cols = [c for c in ["audio_id", "language", "subset", "question", "reference", "url",
                        "model_answer", "requested_model", "model_id"] if c in samp.columns]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    samp[cols].to_csv(args.out, index=False)
    errs = int(samp["model_answer"].astype(str).str.startswith("ERROR").sum())
    print(f"wrote {args.out}: {count} distractors ({args.frac}*{args.models}*{base}), {errs} errors")


if __name__ == "__main__":
    main()
