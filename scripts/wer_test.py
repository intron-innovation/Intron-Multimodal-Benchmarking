"""
Simple jiwer WER check for the Sahara transcription — with and without normalization.
Edit SUBSET / the normalize() rules and re-run to test for yourself.

Run:
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/wer_test.py
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/wer_test.py fulfulde
"""
import re
import sys

import jiwer
import pandas as pd

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "good_english"
CSV = f"outputs/gates_transcription/sahara_{SUBSET}.csv"

df = pd.read_csv(CSV)
refs = df["reference"].astype(str).tolist()     # ground truth
hyps = df["hypothesis"].astype(str).tolist()    # Sahara ASR output
print(f"file: {CSV}   ({len(refs)} utterances)\n")


# --- normalization: lowercase + strip punctuation + collapse whitespace -------
def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)      # drop punctuation (unicode-aware)
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


# --- WITHOUT normalization: jiwer's own default (case, punctuation, all kept) --
wer_raw = jiwer.wer(refs, hyps)

# --- WITH normalization -------------------------------------------------------
wer_norm = jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps])

print(f"WER WITHOUT normalization (jiwer default) : {wer_raw*100:6.2f}%")
print(f"WER WITH    normalization (case+punct+ws) : {wer_norm*100:6.2f}%")

# --- peek at one example so you can see what changed --------------------------
i = 0
print("\n--- example utterance 0 ---")
print("REF raw :", refs[i][:90])
print("HYP raw :", hyps[i][:90])
print("REF norm:", normalize(refs[i])[:90])
print("HYP norm:", normalize(hyps[i])[:90])
