"""
Transcribe Fulfulde via the Intron Voice STT file-upload API (async), to add as a
third ASR ("intron") in the ASR benchmark alongside sahara/omni.

Flow:  POST /file/v1/upload (multipart, use_language_asr_input=ff) -> data.file_id
       GET  /file/v1/status/{file_id} -> poll until data.processing_status==FILE_TRANSCRIBED
       transcript is at data.audio_transcript

API is rate-limited to ~1 request/second, so calls are spaced >=1.1s.

Run:  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/transcribe_intron_api.py \
        --in outputs/gates_transcription/sahara_fulfulde.csv --lang ff --subset fulfulde \
        --out outputs/gates_transcription_intron/intron_fulfulde.csv
"""
import argparse
import os
import time

import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://infer.voice.intron.io/file/v1"
GAP = 1.15  # seconds between API calls (rate limit is 1/sec)


def load_key():
    for line in open(os.path.join(REPO, ".env")):
        if line.strip().lower().startswith("intron_key"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("intron_key not found in .env")


def api_get(url, headers, params=None, tries=6):
    for i in range(tries):
        r = requests.get(url, headers=headers, params=params, timeout=90)
        j = r.json()
        if j.get("status") == "Error" and "ratelimit" in str(j.get("message", "")).lower():
            time.sleep(GAP + 0.5 * i)
            continue
        return j
    return {"data": {}, "status": "Error", "message": "rate-limited"}


def upload(path, lang, headers):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/upload", headers=headers,
                          data={"audio_file_name": os.path.basename(path),
                                "use_language_asr_input": lang},
                          files={"audio_file_blob": f}, timeout=180)
    return r.json().get("data", {}).get("file_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="csv with audio_id, local_path, reference, url")
    ap.add_argument("--lang", default="ff")
    ap.add_argument("--subset", default="fulfulde")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output (default: skip to protect prior transcriptions)")
    args = ap.parse_args()

    outp = os.path.join(REPO, args.out)
    if os.path.exists(outp) and not args.force:
        print(f"skip (exists, use --force to overwrite): {outp}")
        return

    headers = {"Authorization": f"Bearer {load_key()}"}
    d = pd.read_csv(os.path.join(REPO, args.infile))
    if args.limit:
        d = d.head(args.limit)
    d = d.reset_index(drop=True)

    # 1) upload all, collect file_ids
    fid = {}
    for _, r in d.iterrows():
        lp = r["local_path"]
        if not (isinstance(lp, str) and os.path.exists(lp)):
            print(f"  ! missing audio: {r['audio_id']}"); continue
        fid[r["audio_id"]] = upload(lp, args.lang, headers)
        time.sleep(GAP)
    print(f"uploaded {len(fid)} files")

    # 2) poll until all done
    result, pending = {}, dict(fid)
    deadline = time.time() + 60 * 30
    while pending and time.time() < deadline:
        for aid, f in list(pending.items()):
            j = api_get(f"{BASE}/status/{f}", headers, {"get_structured_post_processing": "t"})
            st = j.get("data", {}).get("processing_status")
            if st == "FILE_TRANSCRIBED":
                result[aid] = j["data"].get("audio_transcript", "")
                del pending[aid]
            elif st == "FILE_PROCESSING_FAILED":
                result[aid] = "ERROR: FILE_PROCESSING_FAILED"
                del pending[aid]
            time.sleep(GAP)
        if pending:
            print(f"  waiting on {len(pending)} ...")

    # 3) write out (same shape as sahara/omni transcription files)
    out = d[["audio_id", "reference", "url"]].copy()
    out["language"] = args.subset
    out["subset"] = args.subset
    out["hypothesis"] = out["audio_id"].map(result)
    out = out[["audio_id", "language", "subset", "reference", "hypothesis", "url"]]
    outp = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    out.to_csv(outp, index=False)
    errs = int(out["hypothesis"].astype(str).str.startswith("ERROR").sum())
    miss = int(out["hypothesis"].isna().sum())
    print(f"wrote {outp}: {len(out)} rows, {errs} failed, {miss} missing")


if __name__ == "__main__":
    main()
