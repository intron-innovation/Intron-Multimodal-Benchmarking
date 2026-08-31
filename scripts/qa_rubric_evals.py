"""
Spoken QA rubric evaluation: dimension tables and judge-vs-human agreement.
==========================================================================

Turns per-answer rubric scores into the two things the benchmark reports for
Spoken QA beyond COMET:

1. **Dimension tables** — mean score per language x model for each of the 13
   dimensions, written one file per dimension so they read like the WER/CER
   and BLEU/chrF/COMET tables:

       evaluations/spoken_qa/spoken_qa_factuality.csv
       evaluations/spoken_qa/spoken_qa_harm.csv
       ...
       evaluations/spoken_qa/spoken_qa_rubric_overall.csv   (mean of all 13)

2. **Judge-vs-human agreement** — how closely an LLM judge
   (``qa_rubric_judge.py``) reproduces the physician expert panel, per
   dimension, plus the panel's own inter-rater reliability. No judge score
   belongs in a results table without this:

       evaluations/spoken_qa/judge_agreement_<judge>.csv
       evaluations/spoken_qa/judge_overall_<judge>.csv
       evaluations/spoken_qa/rater_agreement.csv
       evaluations/spoken_qa/summary_<judge>.txt

Scales
------
Every score is put on the rubric's NATURAL direction before anything is
computed: 5 = most of the label, so 5 = most harmful on ``harm`` and 5 = most
fabricated on ``hallucination``. The expert-panel export already stores its
ratings that way; a judge CSV written on the "5 = best on every column" scale
is converted on load via ``--judge-scale best5``. Comparing the two scales
without converting is what makes a judge that agrees with the panel on harm
and hallucination look like it anti-correlates with it.

The four negative dimensions are flipped to good-direction only for the
``spoken_qa_rubric_overall`` table, where a single mean across all 13
dimensions would otherwise mix directions.

Input
-----
* ``--human``  expert-panel ratings, one row per (answer, rater). Scores are
  averaged across raters per ``answer_id`` before anything else.
* ``--judge-scores``  optional per-answer judge scores for the same answers;
  only answers present in both files are compared. Answers are matched on
  ``answer_id`` + ``model`` + a hash of the answer text: ``answer_id`` alone
  is NOT unique — the same id appears once per model that answered the
  question, and a couple of ids carry more than one answer text — so joining
  on it averages one model's rating onto another model's answer. A judge CSV
  carries the panel's own long-named rating columns through verbatim, so the
  judge's canonical column always wins on load — otherwise the "agreement"
  computed for seven of the dimensions is the panel against itself.

Usage:
    python scripts/qa_rubric_evals.py --human "data/Spoken QA/expert_panel_ratings.csv"
    python scripts/qa_rubric_evals.py --human "data/Spoken QA/expert_panel_ratings.csv" \
        --judge-scores results/spoken_qa_rubric/claude_scores.csv --judge claude
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from qa_rubric import HUMAN_RATING_COLUMNS, NEGATIVE_DIMS, answer_key

OUT_DIR = "evaluations/spoken_qa"

SCORE_COLS = HUMAN_RATING_COLUMNS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ratings(path: str, label: str, modality: str | None = "audio",
                 scale: str = "natural") -> pd.DataFrame:
    """
    Load a ratings CSV, rename the rubric columns to canonical labels and put
    the scores on the rubric's natural direction.

    Spoken QA is the ``modality == "audio"`` slice: answers the model produced
    from the spoken question rather than from a transcript or an image.

    A judge CSV preserves its input columns verbatim, so it can hold both the
    panel's long-named rating (``"expert recall"``) and the judge's own score
    (``expert_recall``). Where both exist the canonical column is the judge's
    score and the long one is dropped.
    """
    print(f"Loading {label}: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    print(f"  Shape: {df.shape}")
    if "answer_id" not in df.columns:
        raise ValueError(f"{label} CSV must contain an 'answer_id' column.")

    if modality and "modality" in df.columns:
        df = df[df["modality"] == modality]
        print(f"  modality == {modality!r}: {len(df)} rows")

    rename, shadowed = {}, []
    for canonical, src in SCORE_COLS.items():
        if src == canonical or src not in df.columns:
            continue
        if canonical in df.columns:
            shadowed.append(src)      # canonical column already holds this file's score
        else:
            rename[src] = canonical
    if shadowed:
        print(f"  NOTE: {len(shadowed)} rating columns carried over from the input "
              f"are shadowed by this file's own scores and dropped: {shadowed}")
        df = df.drop(columns=shadowed)
    df = df.rename(columns=rename)

    if scale == "best5":
        flip = [d for d in NEGATIVE_DIMS if d in df.columns]
        df[flip] = 6 - df[flip]
        print(f"  Converted {len(flip)} negative dimensions from 5=best to natural direction")

    missing = [d for d in SCORE_COLS if d not in df.columns]
    if missing:
        print(f"  NOTE: {len(missing)} dimensions absent and will be skipped: {missing}")
    return df


def present_dims(df: pd.DataFrame) -> list[str]:
    return [d for d in SCORE_COLS if d in df.columns]


# ---------------------------------------------------------------------------
# 1. Dimension tables (language x model, one file per dimension)
# ---------------------------------------------------------------------------

def dimension_tables(df: pd.DataFrame, out_dir: str = OUT_DIR,
                     prefix: str = "spoken_qa") -> None:
    """
    Mean rubric score per language x model, one CSV per dimension.

    Rows are languages and columns are answering models, matching the layout
    of the transcription and translation evaluation tables.
    """
    dims = present_dims(df)
    if not dims or "language" not in df.columns or "model" not in df.columns:
        print("Skipping dimension tables: need 'language', 'model' and score columns.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for dim in dims:
        table = (df.pivot_table(index="language", columns="model", values=dim,
                                aggfunc="mean")
                   .rename_axis(index="Language", columns=None)
                   .sort_index()
                   .round(3)
                   .reset_index())
        path = os.path.join(out_dir, f"{prefix}_{dim}.csv")
        table.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Overall = mean across all 13 dimensions, per answer, then averaged.
    # Negative dimensions are flipped to good-direction first (6 - score) so
    # the mean is not the average of two opposing scales; higher = better.
    overall = df.copy()
    for dim in dims:
        if dim in NEGATIVE_DIMS:
            overall[dim] = 6 - overall[dim]
    overall["rubric_overall"] = overall[dims].mean(axis=1)
    table = (overall.pivot_table(index="language", columns="model",
                                 values="rubric_overall", aggfunc="mean")
                    .rename_axis(index="Language", columns=None)
                    .sort_index()
                    .round(3)
                    .reset_index())
    path = os.path.join(out_dir, f"{prefix}_rubric_overall.csv")
    table.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Judge-vs-human agreement
# ---------------------------------------------------------------------------

def krippendorff_alpha(data: pd.DataFrame) -> float:
    """Ordinal Krippendorff's alpha. data: rows=subjects, cols=raters."""
    data = data.dropna(how="all")
    if data.shape[1] < 2:
        return float("nan")
    values = data.values.astype(float)

    o_sum, n_pairs = 0.0, 0
    for row in values:
        valid = row[~np.isnan(row)]
        m = len(valid)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                o_sum += (valid[a] - valid[b]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return float("nan")

    all_vals = values[~np.isnan(values)]
    val_counts = {}
    for v in all_vals:
        val_counts[v] = val_counts.get(v, 0) + 1
    total = len(all_vals)
    e_sum = 0.0
    for a, ca in val_counts.items():
        for b, cb in val_counts.items():
            e_sum += ca * cb * (a - b) ** 2
    e_sum /= total * (total - 1)

    if e_sum == 0:
        return float("nan")
    return 1 - (o_sum / n_pairs) / e_sum


def aggregate_human_scores(human_df: pd.DataFrame) -> pd.DataFrame:
    """Average human scores per answer across all raters."""
    dims = present_dims(human_df)
    keyed = human_df.assign(_key=answer_key(human_df))
    agg = keyed.groupby("_key")[dims].agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_raters"] = keyed.groupby("_key").size()
    return agg


def compute_rater_agreement(human_df: pd.DataFrame) -> pd.DataFrame:
    """Krippendorff alpha per dimension across all human raters."""
    if "user_id" not in human_df.columns:
        print("  NOTE: no 'user_id' column — skipping inter-rater reliability.")
        return pd.DataFrame()

    keyed = human_df.assign(_key=answer_key(human_df))
    records = []
    for dim in present_dims(human_df):
        pivot = keyed.pivot_table(index="_key", columns="user_id",
                                  values=dim, aggfunc="first")
        alpha = krippendorff_alpha(pivot)
        records.append({
            "dimension":          dim,
            "n_answers":          pivot.shape[0],
            "n_raters_total":     pivot.shape[1],
            "krippendorff_alpha": round(alpha, 3) if not np.isnan(alpha) else None,
        })
    return pd.DataFrame(records)


def compute_correlations(judge_df: pd.DataFrame, human_agg: pd.DataFrame,
                         judge_name: str):
    """
    Per-dimension and overall agreement between judge scores and the mean
    human rating. Only answer_ids present in both frames are used.
    """
    judge_indexed = (judge_df.assign(_key=answer_key(judge_df))
                             .drop_duplicates(subset="_key")
                             .set_index("_key"))
    common_ids = judge_indexed.index.intersection(human_agg.index)
    print(f"  Overlap: {len(common_ids)} answers "
          f"(judge: {len(judge_indexed)}, human-rated: {len(human_agg)})")

    if len(common_ids) == 0:
        print("  ERROR: no overlapping answers — check that both files share "
              "the same answer_id, model and answer text.")
        return pd.DataFrame(), {}

    merged = judge_indexed.loc[common_ids].join(human_agg, how="inner")

    results = []
    for dim in SCORE_COLS:
        human_col = f"{dim}_mean"
        if dim not in merged.columns or human_col not in merged.columns:
            continue

        valid = merged[dim].notna() & merged[human_col].notna()
        n = int(valid.sum())
        if n < 5:
            continue

        jv = merged.loc[valid, dim].values.astype(float)
        hv = merged.loc[valid, human_col].values.astype(float)
        pr, pp = stats.pearsonr(jv, hv)
        sr, sp = stats.spearmanr(jv, hv)

        results.append({
            "judge":       judge_name,
            "dimension":   dim,
            "n":           n,
            "pearson_r":   round(pr, 3),
            "pearson_p":   round(pp, 4),
            "spearman_r":  round(sr, 3),
            "spearman_p":  round(sp, 4),
            "mae":         round(float(np.mean(np.abs(jv - hv))), 3),
            "rmse":        round(float(np.sqrt(np.mean((jv - hv) ** 2))), 3),
            "mean_bias":   round(float(np.mean(jv - hv)), 3),
            "judge_mean":  round(float(jv.mean()), 3),
            "human_mean":  round(float(hv.mean()), 3),
        })

    corr_df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)

    # Overall: average all dimensions per answer, then correlate
    dims = [d for d in SCORE_COLS
            if d in merged.columns and f"{d}_mean" in merged.columns]
    merged["judge_overall"] = merged[dims].mean(axis=1)
    merged["human_overall"] = merged[[f"{d}_mean" for d in dims]].mean(axis=1)
    valid = merged["judge_overall"].notna() & merged["human_overall"].notna()
    jv = merged.loc[valid, "judge_overall"].values
    hv = merged.loc[valid, "human_overall"].values

    pr, pp = stats.pearsonr(jv, hv)
    sr, sp = stats.spearmanr(jv, hv)
    overall = {
        "judge":      judge_name,
        "n":          int(valid.sum()),
        "pearson_r":  round(pr, 3),
        "pearson_p":  round(pp, 4),
        "spearman_r": round(sr, 3),
        "spearman_p": round(sp, 4),
        "mae":        round(float(np.mean(np.abs(jv - hv))), 3),
        "rmse":       round(float(np.sqrt(np.mean((jv - hv) ** 2))), 3),
        "mean_bias":  round(float(np.mean(jv - hv)), 3),
    }
    return corr_df, overall


def write_summary(corr_df: pd.DataFrame, overall: dict,
                  n_raters_desc: pd.Series) -> str:
    lines = [
        "=" * 64,
        f"SPOKEN QA RUBRIC — JUDGE vs HUMAN PANEL: {overall['judge']}",
        "=" * 64,
        "",
        f"Answers analysed : {overall['n']}",
        f"Human raters     : {n_raters_desc['mean']:.1f} per answer "
        f"(min {int(n_raters_desc['min'])}, max {int(n_raters_desc['max'])})",
        "",
        "NOTE: Only answer_ids present in BOTH files are included.",
        "      Human scores are averaged across raters before correlating.",
        "",
        "-- OVERALL (mean of all dimensions) ----------------------",
        f"  Pearson r  : {overall['pearson_r']}  (p={overall['pearson_p']})",
        f"  Spearman r : {overall['spearman_r']}  (p={overall['spearman_p']})",
        f"  MAE        : {overall['mae']}",
        f"  RMSE       : {overall['rmse']}",
        f"  Mean bias  : {overall['mean_bias']}  (+ = judge scores higher than humans)",
        "",
        "-- PER-DIMENSION AGREEMENT -------------------------------",
        f"{'Dimension':<28} {'Pearson r':>9} {'Spearman r':>10} {'MAE':>6} {'Bias':>7}",
        "-" * 64,
    ]
    for _, row in corr_df.iterrows():
        sp = "*" if row["pearson_p"] < 0.05 else " "
        ss = "*" if row["spearman_p"] < 0.05 else " "
        lines.append(
            f"{row['dimension']:<28} {row['pearson_r']:>8.3f}{sp}"
            f" {row['spearman_r']:>9.3f}{ss}"
            f" {row['mae']:>6.3f} {row['mean_bias']:>7.3f}"
        )
    lines += ["  * p < 0.05", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Spoken QA rubric scores and check judge-human agreement."
    )
    parser.add_argument("--human", required=True,
                        help="Expert-panel ratings CSV (one row per answer per rater)")
    parser.add_argument("--judge-scores", default=None,
                        help="Per-answer judge scores from qa_rubric_judge.py")
    parser.add_argument("--judge", default=None,
                        help="Judge name for output filenames (default: inferred)")
    parser.add_argument("--human-scale", default="natural", choices=["natural", "best5"],
                        help="Scale of the negative dimensions in the human file "
                             "(default: natural, as exported by the rater UI)")
    parser.add_argument("--judge-scale", default="natural", choices=["natural", "best5"],
                        help="Scale of the negative dimensions in the judge file: "
                             "'natural' for qa_rubric_judge.py defaults, 'best5' for "
                             "CSVs written with --scale best5 (default: natural)")
    parser.add_argument("--modality", default="audio",
                        help="Modality slice to evaluate; '' for all (default: audio)")
    parser.add_argument("--output", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    modality = args.modality or None

    human_df = load_ratings(args.human, "Human ratings", modality, args.human_scale)

    # --- 1. Dimension tables from the human panel ---
    print("\nBuilding dimension tables (language x model)...")
    dimension_tables(human_df, args.output)

    # --- 2. Inter-rater reliability ---
    print("\nComputing inter-rater reliability...")
    agree_df = compute_rater_agreement(human_df)
    if not agree_df.empty:
        path = os.path.join(args.output, "rater_agreement.csv")
        agree_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    if not args.judge_scores:
        print("\nNo --judge-scores supplied; skipping judge agreement.")
        return

    # --- 3. Judge vs human ---
    judge_df = load_ratings(args.judge_scores, "Judge scores", modality, args.judge_scale)
    judge_name = args.judge or os.path.splitext(
        os.path.basename(args.judge_scores))[0].replace("_scores", "")

    n_raters_desc = human_df.groupby(answer_key(human_df)).size().describe()
    print(f"\nHuman raters per answer: {n_raters_desc['mean']:.1f} avg "
          f"(min {int(n_raters_desc['min'])}, max {int(n_raters_desc['max'])})")

    human_agg = aggregate_human_scores(human_df)

    print(f"\nComputing agreement for judge: {judge_name}")
    corr_df, overall = compute_correlations(judge_df, human_agg, judge_name)
    if corr_df.empty:
        print("No results to write.")
        return

    safe = judge_name.replace("/", "_").replace(" ", "_")
    corr_path = os.path.join(args.output, f"judge_agreement_{safe}.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")

    overall_path = os.path.join(args.output, f"judge_overall_{safe}.csv")
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    print(f"  Saved: {overall_path}")

    summary = write_summary(corr_df, overall, n_raters_desc)
    summary_path = os.path.join(args.output, f"summary_{safe}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    print("\n" + summary)


if __name__ == "__main__":
    main()
