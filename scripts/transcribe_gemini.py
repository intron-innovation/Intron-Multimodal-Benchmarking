"""
Transcribe the benchmark audio with Gemini (multimodal ASR) to add as another ASR
in the ASR benchmark alongside sahara/omni/intron.

Feeds each WAV (from the sahara transcription files' local_path) to Gemini with a
verbatim-transcription prompt, in the clip's own language, and writes
outputs/gates_transcription_gemini/gemini_<subset>.csv (same shape as the others).

Run:  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/transcribe_gemini.py \
        --subsets good_english,accented_english,pidgin,hausa,yoruba,fulfulde \
        --model gemini-flash-latest --workers 6
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q  # for get_key + REPO paths

REPO = q.REPO_ROOT
IN_DIR = os.path.join(REPO, "outputs", "gates_transcription")
OUT_DIR = os.path.join(REPO, "outputs", "gates_transcription_gemini")

PROMPT = (
    "Transcribe the following audio exactly as spoken, word for word, in {language}. "
    "The audio is a person speaking {language} (as spoken in Nigeria). "
    "Output ONLY the verbatim transcription text in {language}, using correct spelling and "
    "diacritics/tone marks where applicable. Do not translate to English, do not summarize, "
    "do not add any commentary, labels, or quotation marks."
)


def transcribe(model_id, audio_path, prompt, key):
    from google import genai
    from google.genai import types
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))
        f = client.files.upload(file=audio_path)
        return (client.models.generate_content(model=model_id, contents=[f, prompt]).text or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {str(e)[:160]}"


def run_subset(subset, model_id, key, workers, force):
    out = os.path.join(OUT_DIR, f"gemini_{subset}.csv")
    if os.path.exists(out) and not force:
        print(f"skip (exists): {out}")
        return
    d = pd.read_csv(os.path.join(IN_DIR, f"sahara_{subset}.csv")).reset_index(drop=True)
    lang = str(d["language"].iloc[0]).capitalize()
    prompt = PROMPT.format(language=lang)
    paths = list(d["local_path"])

    def work(p):
        return (transcribe(model_id, p, prompt, key)
                if isinstance(p, str) and os.path.exists(p) else "ERROR: missing audio")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hyps = list(ex.map(work, paths))

    o = d[["audio_id", "reference", "url"]].copy()
    o["language"] = subset
    o["subset"] = subset
    o["hypothesis"] = hyps
    o = o[["audio_id", "language", "subset", "reference", "hypothesis", "url"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    o.to_csv(out, index=False)
    errs = int(o["hypothesis"].astype(str).str.startswith("ERROR").sum())
    print(f"wrote {out}: {len(o)} rows, {errs} errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="good_english,accented_english,pidgin,hausa,yoruba,fulfulde")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    key = q.get_key("google")
    for s in [x.strip() for x in args.subsets.split(",") if x.strip()]:
        run_subset(s, args.model, key, args.workers, args.force)


if __name__ == "__main__":
    main()
