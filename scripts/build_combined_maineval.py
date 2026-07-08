"""
Build the combined ANONYMIZED per-rating table for the main eval (data/main_eval_res/),
mirroring the CHEWS final_table but adapted to our data + cleanups.

Cleanups applied (CHEWS-equivalent where relevant):
  - anonymize: email -> user_id (salted SHA; direct identifiers dropped). A private
    email<->user_id map is written separately.
  - parse `source` -> modality / asr / model ; keep __duplicate flag (is_duplicate)
  - extract the 4 rating criteria from new_text (RAW 1-5; harm kept in original direction)
  - language + output_type (native / english) + a `metadata` tag
  - question / answer lengths
  - low_rater FLAG (not dropped): rater's share of distractor answers scored correct>=3
Not applied (N/A for our data): model-name canonicalization, bad-batch drop, date filter,
category/difficulty enrichment.

Run: /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/build_combined_maineval.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rater_analysis as ra   # parse_source, CRITERIA, CORRECTNESS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "main_eval_res")
OUTDIR = os.path.join(REPO, "data", "rater_tables_main")
OUT = os.path.join(OUTDIR, "final_table_ALL_anon.csv")
MAP = os.path.join(OUTDIR, "email_userid_map_PRIVATE.csv")
SALT_FILE = os.path.join(REPO, "data", ".anon_salt")
LOW_RATER_THRESHOLD = 0.10   # flag if >=10% of a rater's distractors scored correct>=3

CRIT_RAW = {  # criterion text -> raw column name (harm kept in original direction)
    "Answer is correct and consistent with scientific consensus": "factuality",
    "Diagnostics or Management is appropriate for rural African community health setting": "appropriateness",
    "Answer could cause harm or risk safety and wellbeing": "harm",
    "Context is complete and Additional Info is not required OR Context is incomplete and Additional Info is required": "context",
}
DEMOG = ["level", "discipline", "education", "clinical_experience", "age_group", "gender", "country"]


def get_salt():
    if os.path.exists(SALT_FILE):
        return open(SALT_FILE).read().strip()
    salt = hashlib.sha256(os.urandom(32)).hexdigest()
    open(SALT_FILE, "w").write(salt)
    return salt


def lang_of(token):
    if token == "English_Accented": return "Accented English", "native"
    if token == "Fulfulde_English": return "Fulfulde", "english"
    if token == "Hausa_English":    return "Hausa", "english"
    return token, "native"


def main():
    salt = get_salt()
    uid = lambda e: hashlib.sha256((salt + str(e)).encode()).hexdigest()[:40]
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        token = os.path.basename(f).split(" - ")[0] \
            .replace("intron_fresh_audio_Rate_Answers_50_", "").replace("_2026_07_06_batch_None", "")
        lang, otype = lang_of(token)
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            try:
                j = json.loads(r["new_text"]) if pd.notna(r["new_text"]) else {}
            except (json.JSONDecodeError, TypeError):
                j = {}
            mod, asr, model, dup = ra.parse_source(r["source"])
            q = j.get("question", r.get("text")); a = j.get("answer", r.get("prediction"))
            eng = lang in ("English", "Accented English")
            rec = {"email": r.get("email"),  # dropped from output after QC/map
                   "user_id": uid(r.get("email")),
                   "language": lang, "output_type": otype,
                   "modality": mod, "asr": asr, "model": model, "is_duplicate": dup,
                   "metadata": f"{'english' if eng else 'non-english'}_{mod}",
                   "question_length": len(str(q)) if pd.notna(q) else 0,
                   "answer_length": len(str(a)) if pd.notna(a) else 0,
                   "question": q, "answer": a}
            for c in DEMOG:
                rec[c] = r.get(c)
            for crit, name in CRIT_RAW.items():
                rec[name] = pd.to_numeric(j.get(crit), errors="coerce")
            rows.append(rec)
    df = pd.DataFrame(rows)

    # low_rater FLAG (not dropped): share of a rater's DISTRACTOR answers scored factuality>=3
    dist = df[df["model"] == "distractor"]
    rate = dist.groupby("user_id")["factuality"].apply(lambda s: (pd.to_numeric(s, errors="coerce") >= 3).mean())
    df["distractor_correct3plus_rate"] = df["user_id"].map(rate).round(3)
    df["low_rater"] = df["distractor_correct3plus_rate"] >= LOW_RATER_THRESHOLD

    # write private email<->user_id map, then drop the email from the shareable table
    os.makedirs(OUTDIR, exist_ok=True)
    df[["email", "user_id"]].drop_duplicates().to_csv(MAP, index=False)
    df.drop(columns=["email"]).to_csv(OUT, index=False)

    n_low = df.loc[df["low_rater"], "user_id"].nunique()
    print(f"wrote {OUT}")
    print(f"  {len(df)} ratings | {df['user_id'].nunique()} raters | "
          f"{int(df['is_duplicate'].sum())} duplicate rows | {int((df['model']=='distractor').sum())} distractors")
    print(f"  low_rater flagged (>= {LOW_RATER_THRESHOLD:.0%} distractors scored correct>=3, NOT dropped): {n_low} raters")
    print(f"  private map -> {MAP}  (keep out of anything shared)")
    print("  columns:", list(df.drop(columns=['email']).columns))


if __name__ == "__main__":
    main()
