"""
Transcribe the gates_data question sets with the Intron Sahara (NeMo) STT models.

The gates_data CSVs differ from the standard benchmark metadata:
  - audio_path is a remote S3 URL (downloaded + cached locally here)
  - reference text lives in the `new_text` JSON blob (translate files, key
    `translated_text`) or in the plain `text` column (English-only files)
  - the spoken language is implied by the file name

This script builds per-language metadata, downloads audio, maps each language to
the matching <lang>.nemo model file, and transcribes (reusing the long-audio
chunking logic in models/intron_sahara.py). Outputs one CSV per source file to
outputs/gates_transcription/.

Run with the voicebot venv, e.g.:
  INTRON_LOCAL_DIR=/data/avx-voice-bot-engine/models/STT \
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/transcribe_gates.py
"""
import argparse
import json
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlretrieve

import pandas as pd
import torch
import nemo.collections.asr as nemo_asr

# Reuse the long-audio chunking helper from the existing Sahara wrapper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.intron_sahara import transcribe_in_chunks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES_DIR = os.path.join(REPO_ROOT, "data", "gates_data")
AUDIO_CACHE = os.path.join(GATES_DIR, "audio")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "gates_transcription")
STT_DIR = os.environ.get("INTRON_LOCAL_DIR", "/data/avx-voice-bot-engine/models/STT")

# language -> .nemo file stem (without extension)
LANG_TO_MODEL_STEM = {
    "english": "best_tdt_v2_model_sept_30_24",
    "yoruba": "yoruba",
    "hausa": "hausa",
    "pidgin": "pidgin_english",
    "fulfulde": "fulani_fuv_fub_fuq",  # v2.2.1 fulani model (fuv/fub/fuq); prev: fulani.nemo
}

CHUNK_THRESHOLD_SEC = 45.0  # audio longer than this is transcribed in chunks


def classify_file(filename):
    """Return (language, label, ref_key) for a gates_data CSV, or None to skip."""
    name = filename.lower()
    if "eng_to_fulfulde" in name:
        return "fulfulde", "fulfulde", "translated_text"
    if "eng_to_hausa" in name:
        return "hausa", "hausa", "translated_text"
    if "eng_to_pidgin" in name:
        return "pidgin", "pidgin", "translated_text"
    if "eng_to_yoruba" in name:
        return "yoruba", "yoruba", "translated_text"
    if "good_english" in name:
        return "english", "good_english", "text"
    if "heavily_accented_english" in name:
        return "english", "accented_english", "text"
    return None


def get_reference(row, ref_key):
    """Extract the reference transcript for a row."""
    nt = str(row.get("new_text", "") or "").strip()
    if nt:
        try:
            j = json.loads(nt)
            val = j.get(ref_key) or j.get("translated_text") or j.get("text")
            if val:
                return val
        except (json.JSONDecodeError, TypeError):
            pass
    return str(row.get("text", "") or "")


def local_audio_path(url):
    base = os.path.basename(urlparse(url).path)
    return os.path.join(AUDIO_CACHE, base)


def download_audio(url):
    dest = local_audio_path(url)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".tmp"
    try:
        urlretrieve(url, tmp)
        os.replace(tmp, dest)
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"  ! download failed: {url} -> {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def build_metadata(csv_path, language, ref_key, limit=None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    rows = []
    for _, r in df.iterrows():
        url = str(r.get("audio_path", "") or "")
        if not url:
            continue
        rows.append(
            {
                "audio_id": r.get("audio_id", ""),
                "url": url,
                "duration": pd.to_numeric(r.get("audio_duration"), errors="coerce"),
                "reference": get_reference(r, ref_key),
                "language": language,
            }
        )
    return pd.DataFrame(rows)


_MODEL_CACHE = {}


def get_model(language):
    if language in _MODEL_CACHE:
        return _MODEL_CACHE[language]
    stem = LANG_TO_MODEL_STEM.get(language)
    if stem is None:
        print(f"  ! no model mapping for language '{language}', skipping")
        return None
    model_path = os.path.join(STT_DIR, f"{stem}.nemo")
    if not os.path.exists(model_path):
        print(f"  ! model file missing: {model_path} (skipping '{language}')")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading model {model_path} on {device} ...")
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    _disable_cuda_graph_decoder(model)
    _MODEL_CACHE[language] = model
    return model


def _disable_cuda_graph_decoder(model):
    """RNNT/TDT models default to a CUDA-graph greedy decoder that fails on some
    drivers (cudaErrorInsufficientDriver). Disable it for a plain greedy decode."""
    try:
        from omegaconf import open_dict

        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding_cfg)
        print("  cuda-graph greedy decoder disabled")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not adjust decoding strategy: {e})")


def transcribe_df(df, model):
    """Add a 'hypothesis' column. Short clips batched; long clips chunked."""
    df = df.copy()
    df["hypothesis"] = ""
    dur = df["duration"].fillna(0.0)
    short = df[dur <= CHUNK_THRESHOLD_SEC]
    long = df[dur > CHUNK_THRESHOLD_SEC]

    if len(short):
        hyps = model.transcribe(
            short["local_path"].tolist(), batch_size=8, num_workers=2, channel_selector=0
        )
        df.loc[short.index, "hypothesis"] = [h.text.strip() for h in hyps]

    for idx, row in long.iterrows():
        print(f"  chunked ({row['duration']:.0f}s): {os.path.basename(row['local_path'])}")
        df.at[idx, "hypothesis"] = transcribe_in_chunks(
            row["local_path"], os.path.join(REPO_ROOT, "temp_chunks"), model, model_type="nemo"
        )
    return df


def process_file(csv_path, force=False, limit=None):
    filename = os.path.basename(csv_path)
    info = classify_file(filename)
    if info is None:
        print(f"skip (unrecognised): {filename}")
        return
    language, label, ref_key = info
    out_path = os.path.join(OUT_DIR, f"sahara_{label}.csv")
    if os.path.exists(out_path) and not force:
        print(f"skip (exists): {out_path}  (use --force to redo)")
        return

    print(f"\n=== {label}  [{language}]  <- {filename}")
    model = get_model(language)
    if model is None:
        return

    meta = build_metadata(csv_path, language, ref_key, limit=limit)
    print(f"  {len(meta)} rows; downloading audio ...")
    meta["local_path"] = meta["url"].apply(download_audio)
    meta = meta[meta["local_path"].notna()].reset_index(drop=True)
    print(f"  {len(meta)} rows with audio; transcribing ...")

    result = transcribe_df(meta, model)
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["audio_id", "language", "reference", "hypothesis", "duration", "local_path", "url"]
    result[cols].to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", type=str, default="",
        help="comma list to restrict, e.g. yoruba,hausa (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max rows per file (smoke test)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    os.makedirs(AUDIO_CACHE, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    wanted = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    csv_files = sorted(
        os.path.join(GATES_DIR, f) for f in os.listdir(GATES_DIR) if f.endswith(".csv")
    )
    for csv_path in csv_files:
        info = classify_file(os.path.basename(csv_path))
        if info is None:
            continue
        if wanted and info[0] not in wanted:
            continue
        process_file(csv_path, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
