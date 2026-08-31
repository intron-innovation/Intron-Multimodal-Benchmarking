"""
Spoken QA benchmark: the ANSWERING half of the pipeline.
========================================================

Plays each spoken clinical question to a model and records its answer. This
produces the raw predictions; nothing here judges them.

    data/Spoken QA/meta_data.csv
        -> qa_benchmark.py            (this file: answering)
        -> results/spoken_qa/<model>_<language>.csv
        -> evaluations.py             (scoring: COMET)
        -> qa_rubric_judge.py + qa_rubric_evals.py
                                      (scoring: 13 clinical dimensions,
                                       human panel + LLM judge)

Every QA model wrapper has the same signature — ``(input_audio,
input_language, question, output_language) -> {"content": answer}`` — so
models are declared in ``QA_MODELS`` rather than each getting its own
copy-pasted loop.

Usage:
    python scripts/qa_benchmark.py --model gemma4
    python scripts/qa_benchmark.py --model gemini_3_flash --resume
    python scripts/qa_benchmark.py --model gemini_3_flash \
        --model-name gemini-2.0-flash --output-prefix gemini-2.0-flash
"""

import argparse
import importlib
import inspect
import os

import pandas as pd

RESULTS_DIR = "results/spoken_qa"

# Answering models. `output_prefix` is the name that lands in the result
# filename and therefore the model column of every evaluation table, so it
# stays stable even when the wrapper function is renamed.
#
# To add a model: write a wrapper with the signature above (in
# scripts/models/), add an entry here, and add its conda env to the QA block
# of run_benchmarks.sh. Nothing else needs to change.
QA_MODELS = {
    "gemma4": {
        "module": "models.gemma4",
        "function": "spoken_qa",
        "output_prefix": "gemma4",
    },
    "gpt4o_audio_qa": {
        "module": "models.proprietary_models",
        "function": "gpt_4o_audio_qa",
        "output_prefix": "gpt4o-audio-qa",
    },
    "gemini_3_flash": {
        "module": "models.proprietary_models",
        "function": "gemini_qa",
        "output_prefix": "gemini-3-flash-preview",
    },
    "qwen_qa": {
        "module": "models.proprietary_models",
        "function": "qwen_qa",
        "output_prefix": "qwen-plus",
    },
}


def load_answerer(spec: dict):
    """Import the wrapper function for a model, deferring heavy model loads."""
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["function"])


def load_existing_answers(output_prefix: str) -> dict:
    """
    Map audio_path -> answer from any results already on disk.

    A run over a few thousand clips through a rate-limited API rarely
    survives in one go; --resume reuses what previous attempts produced
    instead of paying for it twice.
    """
    answered = {}
    if not os.path.isdir(RESULTS_DIR):
        return answered

    for filename in os.listdir(RESULTS_DIR):
        if not filename.startswith(f"{output_prefix}_") or not filename.endswith(".csv"):
            continue
        prior = pd.read_csv(os.path.join(RESULTS_DIR, filename))
        if "hypothesis" not in prior.columns or "audio_path" not in prior.columns:
            continue
        done = prior[prior["hypothesis"].notna()
                     & (prior["hypothesis"].astype(str).str.strip() != "")
                     & (prior["hypothesis"] != "ERROR")]
        answered.update(dict(zip(done["audio_path"], done["hypothesis"])))
    return answered


def run_inference(df: pd.DataFrame, spec: dict, model_name: str | None = None,
                  output_prefix: str | None = None, resume: bool = False,
                  limit: int | None = None) -> None:
    """Answer every question in `df` and write one CSV per language."""
    answer = load_answerer(spec)
    output_prefix = output_prefix or spec["output_prefix"]

    # Only some wrappers expose the underlying model id (e.g. to run an
    # older Gemini revision); pass it through only where it is accepted.
    extra = {}
    if model_name:
        if "model_name" in inspect.signature(answer).parameters:
            extra["model_name"] = model_name
        else:
            raise ValueError(
                f"{spec['module']}.{spec['function']} has no model_name parameter; "
                f"--model-name cannot be applied to this model.")

    if limit:
        df = df.head(limit)

    df = df.copy()
    df["reference"] = df["answer"]

    existing = load_existing_answers(output_prefix) if resume else {}
    if existing:
        print(f"Resuming: {len(existing)} answers already on disk")

    hypotheses = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        audio_path = row["audio_path"]
        if audio_path in existing:
            hypotheses.append(existing[audio_path])
            continue

        print(f"Processing {position}/{len(df)}: {audio_path}")
        result = answer(audio_path, row["language"], row["question"], "en", **extra)
        hypotheses.append(result["content"])

    df["hypothesis"] = hypotheses

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for language, group in df.groupby("language"):
        path = os.path.join(RESULTS_DIR, f"{output_prefix}_{language}.csv")
        group.to_csv(path, index=False)
        failed = (group["hypothesis"] == "ERROR").sum()
        print(f"  Saved: {path}  ({len(group)} answers"
              + (f", {failed} failed)" if failed else ")"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Spoken QA answers for one model.")
    parser.add_argument("--model", required=True, choices=sorted(QA_MODELS),
                        help="Model to answer with")
    parser.add_argument("--model-name", default=None,
                        help="Override the provider's model id (only for wrappers "
                             "that accept one, e.g. gemini_3_flash)")
    parser.add_argument("--output-prefix", default=None,
                        help="Filename/model label for the results "
                             "(default: the model's registered prefix). Set this "
                             "when --model-name changes what actually answered.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse answers already present in results/spoken_qa/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Answer only the first N questions (smoke test)")
    args = parser.parse_args()

    df = pd.read_csv("data/Spoken QA/meta_data.csv")

    # Audio paths in the metadata are relative to data/
    cwd = os.getcwd()
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(cwd, "data", x))
    df["file_exists"] = df["audio_path"].apply(os.path.exists)
    print(df["file_exists"].value_counts())
    df = df[df["file_exists"]]

    run_inference(df, QA_MODELS[args.model], args.model_name,
                  args.output_prefix, args.resume, args.limit)


if __name__ == "__main__":
    main()
