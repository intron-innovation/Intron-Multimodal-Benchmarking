"""
Combine the per-model QA outputs into datasets keyed by language (subset) and
modality, stacking all models with an added `model` column.

Outputs to results/combined/:
  <modality>_<subset>.csv   one file per (modality, language) with every model stacked
  <modality>_all.csv        all languages for a modality
  qa_all.csv                everything (both modalities)
"""
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "combined")
# (modality, asr) -> source dir. asr = which transcription fed the text; audio has none.
SOURCES = {
    ("text", "sahara"): "results/text_qa",
    ("text", "omni"): "results/text_qa_omni",
    ("audio", "none"): "results/audio_qa",
}
# unified column order (question only exists for text; local_path only for audio)
COLS = ["model", "modality", "asr", "language", "subset", "audio_id",
        "question", "reference", "model_answer", "model_id", "url"]


# models to leave out of the combined outputs (raw per-model files are untouched)
EXCLUDE_MODELS = {"gemma4-e4b"}


def load(modality, asr, d):
    frames = []
    for f in sorted(glob.glob(os.path.join(REPO, d, "*.csv"))):
        base = os.path.basename(f)[:-4]
        model, subset = base.split("_", 1)          # labels use hyphens, subset after 1st _
        if model in EXCLUDE_MODELS:
            continue
        df = pd.read_csv(f)
        df["model"] = model
        df["modality"] = modality
        df["asr"] = asr
        df["subset"] = subset
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[COLS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true", help="use *_native source/out dirs")
    args = ap.parse_args()
    out = OUT
    sources = SOURCES
    if args.native:
        out = os.path.join(REPO, "results", "combined_native")
        sources = {
            ("text", "sahara"): "results/text_qa_native",
            ("text", "omni"): "results/text_qa_omni_native",
            ("audio", "none"): "results/audio_qa_native",
        }
    os.makedirs(out, exist_ok=True)
    by_modality = {}
    for (modality, asr), d in sources.items():
        big = load(modality, asr, d)
        if big.empty:
            continue
        by_modality.setdefault(modality, []).append(big)
        print(f"{modality}/{asr}: {big['model'].nunique()} models x "
              f"{big['subset'].nunique()} languages = {len(big)} rows")

    all_frames = []
    for modality, parts in by_modality.items():
        big = pd.concat(parts, ignore_index=True)
        all_frames.append(big)
        # per (modality, language/subset) — stacks all ASR sources + models
        for subset, g in big.groupby("subset"):
            g.to_csv(os.path.join(out, f"{modality}_{subset}.csv"), index=False)
        big.to_csv(os.path.join(out, f"{modality}_all.csv"), index=False)

    if all_frames:
        master = pd.concat(all_frames, ignore_index=True)
        master.to_csv(os.path.join(out, "qa_all.csv"), index=False)
        print(f"master qa_all.csv: {len(master)} rows")


if __name__ == "__main__":
    main()
