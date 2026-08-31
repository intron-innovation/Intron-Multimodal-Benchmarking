"""
Spoken QA rubric scoring: LLM-as-judge across the 13 clinical dimensions.
========================================================================

Runs an LLM judge over Spoken QA answers and scores each one on the 13-point
rubric defined in ``qa_rubric.py`` — the same rubric the physician expert
panel used. This is the scalable stand-in for the human panel: the judge is
validated against the panel's ratings by ``qa_rubric_evals.py``, which
reports per-dimension Pearson/Spearman agreement before any judge score is
used in place of a human one.

Relationship to the other Spoken QA metric
------------------------------------------
``evaluations.py::spoken_qa_evals`` computes COMET between the predicted and
reference answer — a single semantic-similarity number. It cannot tell a
locally infeasible answer from a feasible one, or a dangerous answer from a
safe one. The dimension scores produced here are what the benchmark reports
for factuality, appropriatness, adequacy, harm, hallucination, and the rest.

Input
-----
Any CSV with ``scenario``, ``question``, ``answer`` columns, plus (optionally)
``answer_id``, ``model``, ``language`` and ``modality`` for filtering and for
joining back to human ratings:

* expert-panel rows (``modality == "audio"``) — re-scores answers that
  physicians have already rated, so judge and human can be compared
* ``results/spoken_qa/<model>_<language>.csv`` — benchmark predictions, where
  the model answer lives in ``hypothesis`` (use ``--answer-col hypothesis``)

Output
------
``results/spoken_qa_rubric/<judge>_scores.csv`` — every input column preserved
verbatim, with the 13 score columns appended on the right, in the natural
direction of each rubric label (5 = most of the label, so 5 = most harmful on
``harm``). That is the scale the expert panel's ratings use, so judge and
human columns can be compared without any further conversion. Pass
``--scale best5`` to invert the four negative dimensions instead.

Judges
------
    claude    claude-opus-4-7            ANTHROPIC_API_KEY
    gpt       gpt-5.5                    OPENAI_API_KEY
    qwen      qwen3.6-plus (DashScope)   DASHSCOPE_API_KEY
    deepseek  deepseek-reasoner          DEEPSEEK_API_KEY

Requirements:
    pip install anthropic openai pandas python-dotenv

Usage:
    export ANTHROPIC_API_KEY="..."
    python scripts/qa_rubric_judge.py --csv "data/Spoken QA/expert_panel_ratings.csv" \
        --judge claude --modality audio
    python scripts/qa_rubric_judge.py --csv "data/Spoken QA/expert_panel_ratings.csv" \
        --judge claude --modality audio --max-rows 10   # test run
"""

import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

from qa_rubric import (
    DIMS,
    DIM_LABELS,
    answer_key,
    build_example_pool,
    build_rubric_preamble,
    build_scoring_request,
    extract_json,
    normalise_scores,
    sample_examples,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------

N_EXAMPLES_POOL = 10          # rows reserved as the few-shot example pool
N_EXAMPLES_PER_CALL = (3, 5)  # (min, max) examples sampled per API call

# provider: "anthropic" uses the Anthropic SDK; "openai" covers every
# OpenAI-compatible endpoint (OpenAI, DashScope, DeepSeek).
JUDGES = {
    "claude": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "gpt": {
        "provider": "openai",
        "model": "gpt-5.5",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "qwen": {
        "provider": "openai",
        "model": "qwen3.6-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # Qwen3 emits <think> blocks unless thinking is disabled; extract_json
        # copes with either, but skipping the CoT is cheaper.
        "extra_body": {"enable_thinking": False},
    },
    "deepseek": {
        "provider": "openai",
        "model": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
}


def build_client(cfg: dict):
    """Instantiate the SDK client for a judge config."""
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise ValueError(
            f"{cfg['api_key_env']} not set.\n"
            f"Run: export {cfg['api_key_env']}='your-key-here'"
        )

    if cfg["provider"] == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


class Usage:
    """Running token totals for a scoring run, so the bill is visible."""

    def __init__(self):
        self.input = self.cached = self.cache_write = self.output = 0
        self.calls = 0

    def add(self, response) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        self.calls += 1
        self.input += getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", 0) or 0
        self.cached += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def report(self) -> str:
        cache_note = ""
        if self.cached or self.cache_write:
            share = self.cached / max(1, self.input + self.cached) * 100
            cache_note = (f"  cache: {self.cached:,} read ({share:.0f}% of input), "
                          f"{self.cache_write:,} written\n")
        return (f"\nTokens over {self.calls:,} calls:\n"
                f"  input (uncached): {self.input:,}\n"
                f"{cache_note}"
                f"  output: {self.output:,}")


def call_judge(client, cfg: dict, preamble: str, request: str,
               usage: Usage | None = None) -> str:
    """
    Send one scoring prompt and return the raw response text.

    The rubric preamble goes in the system slot and the per-answer request in
    the user message. On Anthropic that lets the preamble — ~80% of the
    prompt, identical on every call — be served from cache after the first
    request; the OpenAI-compatible endpoints apply their own automatic prefix
    caching to the same stable prefix.
    """
    if cfg["provider"] == "anthropic":
        kwargs = {
            "model": cfg["model"],
            "max_tokens": cfg.get("max_tokens", 4096),
            "system": [{
                "type": "text",
                "text": preamble,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": request}],
        }
        if cfg.get("effort"):
            kwargs["output_config"] = {"effort": cfg["effort"]}
        response = client.messages.create(**kwargs)
        if usage:
            usage.add(response)
        # Adaptive thinking can precede the answer; take the text blocks only.
        return "".join(b.text for b in response.content if b.type == "text")

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": preamble},
            {"role": "user", "content": request},
        ],
        extra_body=cfg.get("extra_body"),
    )
    if usage:
        usage.add(response)
    message = response.choices[0].message
    raw = message.content or ""
    # Some reasoning models put everything in reasoning_content instead
    if not raw.strip():
        raw = getattr(message, "reasoning_content", "") or ""
    return raw


def score_answer(client, cfg: dict, scenario: str, question: str, answer: str,
                 example_pool: list, n_examples_range: tuple = N_EXAMPLES_PER_CALL,
                 retries: int = 3, invert_negative: bool = False,
                 pool_inverted: bool = False, preamble: str | None = None,
                 usage: Usage | None = None) -> dict | None:
    """Score one answer on all 13 dimensions. Returns None if every retry fails."""
    examples = sample_examples(example_pool, *n_examples_range)
    request = build_scoring_request(scenario, question, answer, examples, pool_inverted)
    preamble = build_rubric_preamble() if preamble is None else preamble

    for attempt in range(retries):
        try:
            raw = call_judge(client, cfg, preamble, request, usage)
            return normalise_scores(extract_json(raw), invert_negative)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            # Rate limits surface differently per SDK; back off on anything
            # that isn't a parse error and retry.
            wait = 2 ** attempt * 5 if "rate" in str(e).lower() else 2
            print(f"    Error (attempt {attempt + 1}): {e} — waiting {wait}s")
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def select_rows(df: pd.DataFrame, modality: str | None, language: str | None,
                answer_model: str | None, answer_col: str,
                dedupe: bool = True) -> pd.DataFrame:
    """
    Filter to the answers to be scored and expose them under an ``answer``
    column, which is what the rubric prompt expects.

    Spoken QA is the ``modality == "audio"`` slice of the expert-panel file;
    benchmark prediction files carry the model answer in ``hypothesis``.

    The panel export holds one row per (answer, rater), so the same answer
    recurs once per physician who rated it. ``dedupe`` keeps one row per
    answer — scoring the duplicates would multiply the API bill and produce
    conflicting scores for the same text.
    """
    if modality and "modality" in df.columns:
        df = df[df["modality"] == modality]
        print(f"  modality == {modality!r}: {len(df)} rows")
    if language and "language" in df.columns:
        df = df[df["language"] == language]
        print(f"  language == {language!r}: {len(df)} rows")
    if answer_model and "model" in df.columns:
        df = df[df["model"] == answer_model]
        print(f"  model == {answer_model!r}: {len(df)} rows")

    if answer_col != "answer":
        if answer_col not in df.columns:
            raise ValueError(f"--answer-col {answer_col!r} not found in the CSV.")
        # Keep the reference answer around; the judge rates the prediction.
        df = df.rename(columns={"answer": "reference_answer", answer_col: "answer"})

    df = df[df["answer"].notna() & (df["answer"].astype(str).str.strip() != "")]

    if dedupe and "answer_id" in df.columns:
        before = len(df)
        df = df.assign(_key=answer_key(df)).drop_duplicates("_key").drop(columns="_key")
        if before != len(df):
            print(f"  deduplicated {before} rating rows to {len(df)} unique answers")

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Take ``n`` answers spread evenly over language x model.

    A plain random sample follows the corpus imbalance (English and the
    talkative models dominate), which leaves the smaller cells too thin to
    report a per-language mean.
    """
    strata = [c for c in ("language", "model") if c in df.columns]
    if not strata:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    groups = df.groupby(strata)
    per_group = max(1, n // groups.ngroups)
    # Cells are uneven (Igbo has a handful of answers, English hundreds), so
    # each contributes at most what it has.
    picked = [g.sample(n=min(len(g), per_group), random_state=seed).index
              for _, g in groups]
    sampled = df.loc[[i for idx in picked for i in idx]]

    # Top up from what is left if the even split undershoots n
    if len(sampled) < n:
        rest = df.drop(index=sampled.index)
        top_up = rest.sample(n=min(len(rest), n - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, top_up])

    sampled = sampled.head(n).reset_index(drop=True)
    print(f"  stratified sample: {len(sampled)} answers over "
          f"{groups.ngroups} {' x '.join(strata)} cells")
    return sampled



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score Spoken QA answers on the 13-dimension clinical rubric."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--judge", default="claude", choices=sorted(JUDGES),
                        help="Judge model to score with (default: claude)")
    parser.add_argument("--judge-model", default=None,
                        help="Override the judge's model id (e.g. claude-opus-4-7 to "
                             "match an earlier run)")
    parser.add_argument("--effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort for judges that support it. Scoring "
                             "against a fixed rubric is a bounded task; low keeps "
                             "thinking tokens down (default: low)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: results/spoken_qa_rubric/<judge>_scores.csv)")
    parser.add_argument("--answer-col", default="answer",
                        help="Column holding the answer to score "
                             "(default: answer; use 'hypothesis' for benchmark outputs)")
    parser.add_argument("--modality", default="audio",
                        help="Keep only rows with this modality; '' to disable (default: audio)")
    parser.add_argument("--language", default=None, help="Keep only this language")
    parser.add_argument("--answer-model", default=None,
                        help="Keep only answers produced by this model")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Score every row, including repeated ratings of the "
                             "same answer (default: one row per unique answer)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Score a random stratified sample of this many answers "
                             "(balanced across language x model)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample (default: 42)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to score (default: all)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--example-pool-size", type=int, default=N_EXAMPLES_POOL,
                        help=f"Rows reserved as few-shot examples (default: {N_EXAMPLES_POOL})")
    parser.add_argument("--examples-per-call", type=int, nargs=2,
                        default=list(N_EXAMPLES_PER_CALL), metavar=("MIN", "MAX"),
                        help="Range of examples sampled per API call (default: 3 5)")
    parser.add_argument("--scale", default="natural", choices=["natural", "best5"],
                        help="Output scale for the 4 negative dimensions: 'natural' "
                             "(5 = most of the label, matches the expert panel) or "
                             "'best5' (inverted, 5 = best on every column). "
                             "Default: natural")
    parser.add_argument("--pool-scale", default=None, choices=["natural", "best5"],
                        help="Scale of the few-shot pool's existing scores "
                             "(default: same as --scale)")
    args = parser.parse_args()

    cfg = dict(JUDGES[args.judge])
    if args.judge_model:
        cfg["model"] = args.judge_model
    if cfg["provider"] == "anthropic":
        cfg["effort"] = args.effort
    output = args.output or f"results/spoken_qa_rubric/{args.judge}_scores.csv"
    invert_negative = args.scale == "best5"
    pool_inverted = (args.pool_scale or args.scale) == "best5"

    # --- Load and filter ---
    print(f"\nLoading {args.csv}...")
    df = pd.read_csv(args.csv, engine="python", on_bad_lines="skip")
    print(f"Total rows: {len(df)}")
    df = select_rows(df, args.modality or None, args.language,
                     args.answer_model, args.answer_col,
                     dedupe=not args.no_dedupe)

    if args.sample and args.sample < len(df):
        df = stratified_sample(df, args.sample, args.seed)
    if "model" in df.columns:
        print(f"Answer models present: {df['model'].value_counts().to_dict()}")

    # Capture the original column order so the output CSV mirrors the input
    # exactly, with score columns appended.
    input_columns = list(df.columns)
    score_columns = [DIM_LABELS[d] for d in DIMS]

    # --- Reserve example pool (excluded from scoring) ---
    example_pool, score_df = build_example_pool(df, args.example_pool_size)

    if args.max_rows:
        score_df = score_df.head(args.max_rows)
        print(f"Limiting scoring to {args.max_rows} rows for this run.")

    client = build_client(cfg)
    preamble = build_rubric_preamble()
    usage = Usage()
    print(f"Judge: {args.judge} ({cfg['model']})  scale: {args.scale}")

    # --- Score each row ---
    results = []
    total = len(score_df)

    for i, row in score_df.iterrows():
        ans_id = str(row.get("answer_id", f"row{i}"))[:20]
        model_name = row.get("model", "unknown")
        print(f"\n[{len(results) + 1}/{total}] answer_id={ans_id}  model={model_name}")

        scores = score_answer(
            client,
            cfg,
            scenario=str(row.get("scenario", "")),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            example_pool=example_pool,
            n_examples_range=tuple(args.examples_per_call),
            invert_negative=invert_negative,
            pool_inverted=pool_inverted,
            preamble=preamble,
            usage=usage,
        )

        if scores:
            # Preserve EVERY input column verbatim, then append score columns.
            record = {col: row[col] for col in input_columns}
            record.update(scores)
            results.append(record)
            print(f"    factuality={scores.get('factuality')}  "
                  f"harm={scores.get('harm')}  "
                  f"hallucination={scores.get('hallucination')}")
        else:
            print("    Failed to score — skipping.")

        time.sleep(args.delay)

    # --- Save ---
    print(usage.report())

    if not results:
        print("\nNo results to save.")
        return

    out_df = pd.DataFrame(results)

    # Enforce column order: original input columns first (unchanged), then the
    # 13 score columns. A score column already present in the input (the human
    # rating) keeps its position and is overwritten by the judge's score.
    appended_scores = [c for c in score_columns if c not in input_columns]
    out_df = out_df[input_columns + appended_scores]

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"\n✓ Saved {len(out_df)} scored rows to: {output}")
    print(f"  (excluded {args.example_pool_size} rows used as few-shot examples)")


if __name__ == "__main__":
    main()
