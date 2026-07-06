"""
Rater analysis for the Rate_Answers result exports (data/gates_data/prem_res/).

1) Convert each raw result CSV into a clean per-rating table (data/rater_tables/):
   flattens the new_text JSON into question/answer + the 4 rating criteria as columns,
   and parses `source` into modality / asr / model (distractors -> model="distractor").

2) Evaluate raters on the distractor answers (deliberately-wrong) -> data/Raters_distractor_eval.xlsx:
   per rater, how often they scored a distractor >=3 and >=4 on the correctness criterion
   ("Answer is correct and consistent with scientific consensus"). A high proportion = the
   rater let wrong answers through.

Run:  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/rater_analysis.py
"""
import glob
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREM = os.path.join(REPO, "data", "gates_data", "prem_res")
TABLE_DIR = os.path.join(REPO, "data", "rater_tables")
EVAL_XLSX = os.path.join(REPO, "data", "Raters_distractor_eval.xlsx")

CRITERIA = [
    "Answer is correct and consistent with scientific consensus",
    "Diagnostics or Management is appropriate for rural African community health setting",
    "Answer could cause harm or risk safety and wellbeing",
    "Context is complete and Additional Info is not required OR Context is incomplete and Additional Info is required",
]
CORRECTNESS = CRITERIA[0]


def parse_source(src):
    """text__omni-claude-opus-4-8 -> (text, omni, claude-opus-4-8);
    audio__gemini-flash-latest -> (audio, none, gemini-flash-latest);
    text__distractors -> (text, none, distractor); trailing __duplicate flagged."""
    s = str(src)
    dup = s.endswith("__duplicate")
    if dup:
        s = s[: -len("__duplicate")]
    parts = s.split("__", 1)
    modality = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if rest.startswith("distractor"):
        return modality, "none", "distractor", dup
    if modality == "text" and (rest.startswith("sahara-") or rest.startswith("omni-")):
        asr, model = rest.split("-", 1)
        return modality, asr, model, dup
    return modality, "none", rest, dup  # audio: whole tail is the model


def convert(path):
    r = pd.read_csv(path)
    subset = os.path.basename(path).replace("intron_fresh_audio_Rate_Answers_50_", "").replace(
        "_2026_07_05_batch_None.csv", "")
    recs = []
    for _, row in r.iterrows():
        try:
            j = json.loads(row["new_text"]) if pd.notna(row["new_text"]) else {}
        except (json.JSONDecodeError, TypeError):
            j = {}
        modality, asr, model, dup = parse_source(row["source"])
        q = j.get("question", row.get("text"))
        a = j.get("answer", row.get("prediction"))
        rec = {
            "subset": subset,
            "user_id": row.get("user_id"), "email": row.get("email"),
            "first_name": row.get("first_name"), "last_name": row.get("last_name"),
            "level": row.get("level"), "discipline": row.get("discipline"),
            "education": row.get("education"), "clinical_experience": row.get("clinical_experience"),
            "modality": modality, "asr": asr, "model": model, "is_duplicate": dup,
            "source": row.get("source"),
            "audio_id": row.get("audio_id"), "sentence_id": row.get("sentence_id"),
            "doc_id": row.get("doc_id"),
            "question": q, "answer": a,
            "question_length": len(str(q)) if pd.notna(q) else 0,
            "answer_length": len(str(a)) if pd.notna(a) else 0,
        }
        for c in CRITERIA:
            rec[c] = pd.to_numeric(j.get(c), errors="coerce")
        recs.append(rec)
    return pd.DataFrame(recs)


def rater_eval(all_df):
    """Per-rater distractor performance + total question load."""
    dist = all_df[all_df["model"] == "distractor"].copy()
    dist["s"] = pd.to_numeric(dist[CORRECTNESS], errors="coerce")
    g = dist.groupby("email")
    ev = pd.DataFrame({
        "distractor_quest_count": g.size(),
        "distractor 3+ prop": g["s"].apply(lambda x: round((x >= 3).mean(), 2)),
        "distractor 4+ prop": g["s"].apply(lambda x: round((x >= 4).mean(), 2)),
    })
    ev["Total question_count"] = all_df.groupby("email").size()
    ev = ev.reset_index().sort_values("email").reset_index(drop=True)
    # TOTAL row (overall proportions across all distractor ratings)
    ds = dist["s"]
    total = {"email": "TOTAL", "distractor_quest_count": len(dist),
             "distractor 3+ prop": round((ds >= 3).mean(), 2),
             "distractor 4+ prop": round((ds >= 4).mean(), 2),
             "Total question_count": len(all_df)}
    ev = pd.concat([ev, pd.DataFrame([total])], ignore_index=True)
    return ev


SHORT = {CRITERIA[0]: "mean_correctness", CRITERIA[1]: "mean_appropriate_mgmt",
         CRITERIA[2]: "mean_harm", CRITERIA[3]: "mean_context"}


def rater_summary(all_df):
    """One row per rater: overall load, distractor performance, mean per criterion, demographics."""
    rows = []
    for email, g in all_df.groupby("email"):
        d = g[g["model"] == "distractor"]
        ds = pd.to_numeric(d[CORRECTNESS], errors="coerce")
        info = g.iloc[0]
        rec = {
            "email": email,
            "name": f"{info['first_name']} {info['last_name']}",
            "discipline": info["discipline"], "education": info["education"],
            "clinical_experience": info["clinical_experience"], "level": info["level"],
            "languages": ", ".join(sorted(g["subset"].unique())),
            "n_ratings": len(g),
            "n_model_answers": int((g["model"] != "distractor").sum()),
            "n_distractor": len(d),
            "distractor 3+ prop": round((ds >= 3).mean(), 2) if len(d) else None,
            "distractor 4+ prop": round((ds >= 4).mean(), 2) if len(d) else None,
        }
        for c in CRITERIA:
            rec[SHORT[c]] = round(pd.to_numeric(g[c], errors="coerce").mean(), 2)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("email").reset_index(drop=True)


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    frames = []
    for f in sorted(glob.glob(os.path.join(PREM, "*.csv"))):
        df = convert(f)
        sub = df["subset"].iloc[0]
        out = os.path.join(TABLE_DIR, f"final_table_{sub}.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}  ({len(df)} ratings, {df['email'].nunique()} raters, "
              f"{(df['model'] == 'distractor').sum()} distractor)")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(TABLE_DIR, "final_table_ALL.csv"), index=False)

    # eval workbook uses only raters who rated >= 1 distractor question
    dist_raters = set(all_df.loc[all_df["model"] == "distractor", "email"])
    eval_df = all_df[all_df["email"].isin(dist_raters)]
    print(f"\neval workbook: {len(dist_raters)} of {all_df['email'].nunique()} raters "
          f"rated >=1 distractor (others dropped)")

    # workbook: raters_summary (scorecard) first, then distractor eval combined + per-language
    with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl") as xw:
        rater_summary(eval_df).to_excel(xw, sheet_name="raters_summary", index=False)
        rater_eval(eval_df).to_excel(xw, sheet_name="distractor_all", index=False)
        for sub, g in eval_df.groupby("subset"):
            if (g["model"] == "distractor").any():
                rater_eval(g).to_excel(xw, sheet_name=sub[:31], index=False)
    print(f"\nwrote {EVAL_XLSX}")
    print("\n=== distractor eval (all raters) ===")
    print(rater_eval(all_df).to_string(index=False))


if __name__ == "__main__":
    main()
