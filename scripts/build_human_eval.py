"""
Transform QA results into the human-eval format (like data/english_audio_human_eval_updated-1.csv):
columns doc_id, sentence_id, text, n_chars, prediction, audio_path, source.

Mapping (per user):
  doc_id, sentence_id  <- from the original gates data (joined on audio_id)
  text                 <- the transcribed text (QA 'question' / STT hypothesis)
  n_chars              <- len(text)
  prediction           <- model_answer
  audio_path           <- JSON {"recorded_audio_path": url, "uploaded_audio_path":"", ...}
  source               <- "{modality}__{model}"
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["doc_id", "sentence_id", "text", "n_chars", "prediction", "audio_path", "source"]
DUP_FRAC = 0.10   # 10% of base questions duplicated, same audio_ids across every source
SEED = 42


def _audio_json(url):
    return json.dumps({"recorded_audio_path": "" if pd.isna(url) else str(url),
                       "uploaded_audio_path": "", "image_path": "", "video_path": "", "pdf_path": ""})


def build(sources, subset, modality, gates_file, out_file, exclude, dfile=None):
    # sources = list of (qa_dir, asr); every model from every ASR source is included
    g = pd.read_csv(gates_file)[["audio_id", "doc_id", "sentence_id"]].drop_duplicates("audio_id")
    lut = g.set_index("audio_id")[["doc_id", "sentence_id"]].to_dict("index")
    rows = []
    for qa_dir, asr in sources:
        for f in sorted(glob.glob(os.path.join(REPO, qa_dir, f"*_{subset}.csv"))):
            model = os.path.basename(f)[: -len(f"_{subset}.csv")]
            # distractors come from --distractors (the 15%*11*base set), not the qa_dir file
            if model in exclude or model == "distractors":
                continue
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                info = lut.get(r["audio_id"], {})
                text = "" if pd.isna(r["question"]) else str(r["question"])
                mid = "" if pd.isna(r.get("model_id")) else str(r.get("model_id"))
                rows.append({
                    "doc_id": info.get("doc_id"),
                    "sentence_id": info.get("sentence_id"),
                    "text": text,
                    "n_chars": len(text),
                    "prediction": r["model_answer"],
                    "audio_path": _audio_json(r["url"]),
                    "source": f"{modality}__{asr}-{mid}",
                    "_aid": r["audio_id"],
                })
    df = pd.DataFrame(rows)

    # duplicates: 10% of base questions (same audio_ids duplicated for every source),
    # tagged with a "__duplicate" source suffix.
    aids = sorted(df["_aid"].dropna().unique())
    k = math.ceil(len(aids) * DUP_FRAC)
    dup_aids = set(np.random.RandomState(SEED).choice(aids, size=k, replace=False))
    dups = df[df["_aid"].isin(dup_aids)].copy()
    dups["source"] = dups["source"] + "__duplicate"
    parts = [df[COLS], dups[COLS]]

    # distractors from the dedicated FRAC*LLMS*base set (source = "<modality>-distractors")
    if dfile and os.path.exists(dfile):
        dd = pd.read_csv(dfile)
        drows = []
        for _, r in dd.iterrows():
            info = lut.get(r["audio_id"], {})
            text = "" if pd.isna(r["question"]) else str(r["question"])
            drows.append({
                "doc_id": info.get("doc_id"), "sentence_id": info.get("sentence_id"),
                "text": text, "n_chars": len(text), "prediction": r["model_answer"],
                "audio_path": _audio_json(r["url"]), "source": f"{modality}__distractors",
            })
        parts.append(pd.DataFrame(drows)[COLS])

    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"wrote {out_file}: {len(out)} rows ({k} questions duplicated per source)")
    print(out["source"].value_counts().to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", required=True, help="comma list, e.g. results/text_qa,results/text_qa_omni")
    ap.add_argument("--subset", required=True, help="e.g. good_english")
    ap.add_argument("--asr", required=True, help="comma list matching qa-dir, e.g. sahara,omni")
    ap.add_argument("--modality", required=True, help="text | audio")
    ap.add_argument("--gates", required=True, help="original gates CSV for the subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="gemma4-e4b")
    ap.add_argument("--distractors", default=None, help="dedicated distractor CSV")
    args = ap.parse_args()
    sources = list(zip([x.strip() for x in args.qa_dir.split(",")],
                       [x.strip() for x in args.asr.split(",")]))
    build(sources, args.subset, args.modality, args.gates, args.out,
          {x.strip() for x in args.exclude.split(",") if x.strip()}, dfile=args.distractors)


if __name__ == "__main__":
    main()
