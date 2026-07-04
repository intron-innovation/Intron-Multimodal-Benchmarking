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


def build(sources, subset, modality, gates_file, out_file, exclude, dfile=None, extra_file=None, text_from=None, only=None):
    # sources = list of (qa_dir, asr); every model from every ASR source is included.
    # audio: no ASR -> source is "audio__<model_id>"; audio QA files lack a transcript, so
    # the `text` field comes from --text-from (the Sahara transcription for the subset).
    g = pd.read_csv(gates_file)[["audio_id", "doc_id", "sentence_id", "audio_path"]].drop_duplicates("audio_id")
    g["audio_id"] = g["audio_id"].astype(str)
    lut = g.set_index("audio_id")[["doc_id", "sentence_id", "audio_path"]].to_dict("index")
    subset_aids = set(g["audio_id"])
    text_lut = {}
    if text_from and os.path.exists(text_from):
        t = pd.read_csv(text_from)
        tcol = "hypothesis" if "hypothesis" in t.columns else "question"
        text_lut = {str(a): ("" if pd.isna(x) else str(x))
                    for a, x in zip(t["audio_id"].astype(str), t[tcol])}
    is_audio = modality == "audio"
    rows = []
    for qa_dir, asr in sources:
        for f in sorted(glob.glob(os.path.join(REPO, qa_dir, f"*_{subset}.csv"))):
            model = os.path.basename(f)[: -len(f"_{subset}.csv")]
            # distractors come from --distractors (the dedicated set), not the qa_dir file
            if model in exclude or model == "distractors":
                continue
            if only and model not in only:   # restrict to an explicit model allow-list
                continue
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                aid = str(r["audio_id"])
                info = lut.get(aid, {})
                qv = r.get("question")
                text = "" if (qv is None or pd.isna(qv)) else str(qv)
                if not text:
                    text = text_lut.get(aid, "")
                mid = "" if pd.isna(r.get("model_id")) else str(r.get("model_id"))
                src = f"{modality}__{mid}" if is_audio else f"{modality}__{asr}-{mid}"
                rows.append({
                    "doc_id": aid,                       # doc_id column carries the audio_id
                    "sentence_id": info.get("sentence_id"),
                    "text": text,
                    "n_chars": len(text),
                    "prediction": r["model_answer"],
                    "audio_path": _audio_json(r["url"]),
                    "source": src,
                    "_aid": aid,
                })

    # extra models from another engineer's combined file (model, response, modality, asr, audio_id, question)
    if extra_file and os.path.exists(extra_file):
        ex = pd.read_csv(extra_file)
        ex = ex[(ex["modality"] == modality) & (ex["audio_id"].astype(str).isin(subset_aids))]
        if not is_audio:  # audio rows carry asr=NaN; text rows filter by the requested ASRs
            asr_set = {a for _, a in sources}
            ex = ex[ex["asr"].isin(asr_set)]
        # dedup so a source-file with duplicate audio rows can't inflate a model's count
        ex = ex.drop_duplicates(["model", "audio_id"] if is_audio else ["model", "asr", "audio_id"])
        for _, r in ex.iterrows():
            if r["model"] in exclude or (only and r["model"] not in only):
                continue
            aid = str(r["audio_id"])
            info = lut.get(aid, {})
            text = "" if pd.isna(r["question"]) else str(r["question"])
            src = f"{modality}__{r['model']}" if is_audio else f"{modality}__{r['asr']}-{r['model']}"
            rows.append({
                "doc_id": aid, "sentence_id": info.get("sentence_id"),
                "text": text, "n_chars": len(text), "prediction": r["response"],
                "audio_path": _audio_json(info.get("audio_path")),
                "source": src,
                "_aid": aid,
            })

    df = pd.DataFrame(rows)

    # duplicates: 10% of base questions (same audio_ids duplicated for every source),
    # tagged with a "__duplicate" source suffix.
    aids = sorted(df["_aid"].dropna().unique())
    k = math.ceil(len(aids) * DUP_FRAC)
    dup_aids = set(np.random.RandomState(SEED).choice(aids, size=k, replace=False))
    dups = df[df["_aid"].isin(dup_aids)].copy()
    dups["source"] = dups["source"] + "__duplicate"
    # text evals omit audio_path; only audio evals carry the recorded-audio reference
    out_cols = COLS if is_audio else [c for c in COLS if c != "audio_path"]
    parts = [df[out_cols], dups[out_cols]]

    # distractors from the dedicated FRAC*LLMS*base set (source = "<modality>-distractors")
    if dfile and os.path.exists(dfile):
        dd = pd.read_csv(dfile)
        drows = []
        for _, r in dd.iterrows():
            aid = str(r["audio_id"])
            info = lut.get(aid, {})
            text = "" if pd.isna(r["question"]) else str(r["question"])
            drows.append({
                "doc_id": aid, "sentence_id": info.get("sentence_id"),
                "text": text, "n_chars": len(text), "prediction": r["model_answer"],
                "audio_path": _audio_json(r["url"]), "source": f"{modality}__distractors",
            })
        parts.append(pd.DataFrame(drows)[out_cols])

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
    ap.add_argument("--extra-file", default=None, help="another engineer's combined results CSV")
    ap.add_argument("--text-from", default=None, help="transcription CSV (audio_id->hypothesis) for the text field (audio)")
    ap.add_argument("--only", default=None, help="comma list of model labels to include (allow-list)")
    args = ap.parse_args()
    sources = list(zip([x.strip() for x in args.qa_dir.split(",")],
                       [x.strip() for x in args.asr.split(",")]))
    only = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    build(sources, args.subset, args.modality, args.gates, args.out,
          {x.strip() for x in args.exclude.split(",") if x.strip()},
          dfile=args.distractors, extra_file=args.extra_file, text_from=args.text_from, only=only)


if __name__ == "__main__":
    main()
