"""
Normalize outputs/gates_transcription/omni_asr_results.csv (OmniASR transcriptions)
into the format qa_text_benchmark.py expects: columns audio_id, language, subset,
hypothesis, reference, url. audio_id / reference / subset (incl. English good vs
accented split) are recovered by joining the OmniASR audio_path basename to the
Sahara transcription files. Writes one file with a `subset` column.
"""
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "outputs", "gates_transcription", "omni_asr_results.csv")
OUT_DIR = os.path.join(REPO, "outputs", "gates_transcription_omni")

# basename(url) -> (audio_id, reference, subset) from the Sahara files
lut = {}
for f in glob.glob(os.path.join(REPO, "outputs", "gates_transcription", "sahara_*.csv")):
    subset = os.path.basename(f)[len("sahara_"):-4]
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        lut[os.path.basename(str(r["url"]))] = (r.get("audio_id"), r.get("reference"), subset, r.get("url"))

omni = pd.read_csv(SRC)
rows, miss = [], 0
for _, r in omni.iterrows():
    base = os.path.basename(str(r["audio_path"]))
    info = lut.get(base)
    if info is None:
        miss += 1
        aid, ref, subset, url = pd.NA, pd.NA, str(r["language"]).replace("Nigerian fulfulde", "fulfulde"), r["audio_path"]
    else:
        aid, ref, subset, url = info
    rows.append({"audio_id": aid, "language": r["language"], "subset": subset,
                 "hypothesis": r["text"], "reference": ref, "url": url})

out = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(os.path.join(OUT_DIR, "omni_all.csv"), index=False)
print(f"{len(out)} rows, {miss} unmatched")
print("per subset:", out["subset"].value_counts().to_dict())
