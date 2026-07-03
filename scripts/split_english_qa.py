"""
Split merged text-QA English outputs (results/text_qa/<model>_english.csv) into
<model>_good_english.csv and <model>_accented_english.csv by audio_id, using the two
source transcription files. No re-inference. Idempotent: only splits files that are
complete (row count == good + accented), leaves partial/in-progress files alone.
"""
import glob
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "outputs", "gates_transcription")
QA = os.path.join(REPO, "results", "text_qa")

good_ids = set(pd.read_csv(os.path.join(IN, "sahara_good_english.csv"))["audio_id"])
acc_ids = set(pd.read_csv(os.path.join(IN, "sahara_accented_english.csv"))["audio_id"])
assert not (good_ids & acc_ids), "audio_id overlap between good and accented English"
expected = len(good_ids) + len(acc_ids)

for f in sorted(glob.glob(os.path.join(QA, "*_english.csv"))):
    # skip already-split files
    if f.endswith("_good_english.csv") or f.endswith("_accented_english.csv"):
        continue
    model = os.path.basename(f)[: -len("_english.csv")]
    d = pd.read_csv(f)
    if len(d) != expected:
        print(f"skip {model}: {len(d)} rows (expected {expected}, not complete)")
        continue
    good = d[d["audio_id"].isin(good_ids)]
    acc = d[d["audio_id"].isin(acc_ids)]
    if len(good) + len(acc) != len(d):
        print(f"WARN {model}: {len(d)} rows -> good {len(good)} + accented {len(acc)} unmatched; leaving as-is")
        continue
    good.to_csv(os.path.join(QA, f"{model}_good_english.csv"), index=False)
    acc.to_csv(os.path.join(QA, f"{model}_accented_english.csv"), index=False)
    os.remove(f)
    print(f"split {model}: good_english={len(good)}, accented_english={len(acc)} (removed merged)")
