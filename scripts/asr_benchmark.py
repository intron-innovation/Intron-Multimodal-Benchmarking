"""
ASR benchmark per language for the Sahara and OmniASR models.

Scores each ASR's hypothesis against the same ground-truth reference (taken from the
Sahara transcription files, joined by audio_id) and reports WER and CER per language.

  Sahara hypotheses: outputs/gates_transcription/sahara_<subset>.csv  (reference, hypothesis)
  Omni   hypotheses: outputs/gates_transcription_omni/omni_all.csv     (subset, hypothesis)

WER = (S + D + I) / N_ref_words  (word level, with substitution/deletion/insertion breakdown)
CER = edit_distance(ref, hyp) / len(ref_chars)  (character level)

Text is normalised (lowercase, punctuation stripped, whitespace collapsed) before scoring;
pass --strip-diacritics to also fold combining marks (e.g. Hausa ƙ/ɗ, Yoruba tone marks).

Run (voicebot venv):
  .../python scripts/asr_benchmark.py
  .../python scripts/asr_benchmark.py --strip-diacritics --out results/asr_benchmark/asr_wer_diacfold.csv
"""
import argparse
import glob
import os
import re
import unicodedata

import pandas as pd
from rapidfuzz.distance import Levenshtein

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAHARA_DIR = os.path.join(REPO, "outputs", "gates_transcription")
OMNI_ALL = os.path.join(REPO, "outputs", "gates_transcription_omni", "omni_all.csv")


def normalize(s, strip_diacritics=False):
    s = "" if pd.isna(s) else str(s)
    s = s.lower().strip()
    if strip_diacritics:
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_edits(ref, hyp):
    """Word-level Levenshtein with S/D/I breakdown (rapidfuzz editops)."""
    r, h = ref.split(), hyp.split()
    S = D = I = 0
    for op in Levenshtein.editops(r, h):
        if op.tag == "replace":
            S += 1
        elif op.tag == "delete":
            D += 1
        else:  # insert
            I += 1
    return S, D, I, len(r)


def edit_distance(a, b):
    """Character-level Levenshtein distance (for CER)."""
    return Levenshtein.distance(a, b)


def load_hyps(strip):
    """Returns {subset: {audio_id: reference}} and {(asr, subset): {audio_id: hyp}}."""
    refs, hyps = {}, {}
    for f in sorted(glob.glob(os.path.join(SAHARA_DIR, "sahara_*.csv"))):
        sub = os.path.basename(f).replace("sahara_", "").replace(".csv", "")
        d = pd.read_csv(f)
        d["audio_id"] = d["audio_id"].astype(str)
        refs[sub] = dict(zip(d["audio_id"], d["reference"]))
        hyps[("sahara", sub)] = dict(zip(d["audio_id"], d["hypothesis"]))
    o = pd.read_csv(OMNI_ALL)
    o["audio_id"] = o["audio_id"].astype(str)
    for sub, g in o.groupby("subset"):
        hyps[("omni", sub)] = dict(zip(g["audio_id"], g["hypothesis"]))
    return refs, hyps


def score(refs, hyps, strip):
    rows = []
    for (asr, sub), hyp_map in sorted(hyps.items()):
        ref_map = refs.get(sub, {})
        ids = [a for a in hyp_map if a in ref_map]
        S = D = I = Nw = 0
        cerr = cchars = 0
        for a in ids:
            ref = normalize(ref_map[a], strip)
            hyp = normalize(hyp_map[a], strip)
            s, dd, ii, n = word_edits(ref, hyp)
            S += s; D += dd; I += ii; Nw += n
            cerr += edit_distance(ref, hyp); cchars += len(ref)
        if Nw == 0:
            continue
        rows.append({"language": sub, "asr": asr, "n_utts": len(ids),
                     "ref_words": Nw, "wer": round((S + D + I) / Nw, 4),
                     "cer": round(cerr / max(cchars, 1), 4),
                     "sub": S, "del": D, "ins": I})
    return pd.DataFrame(rows).sort_values(["language", "asr"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip-diacritics", action="store_true",
                    help="fold combining marks before scoring")
    ap.add_argument("--out", default="results/asr_benchmark/asr_wer_by_language.csv")
    args = ap.parse_args()

    refs, hyps = load_hyps(args.strip_diacritics)
    df = score(refs, hyps, args.strip_diacritics)

    # per-ASR overall (micro-average over all utterances)
    for asr, g in df.groupby("asr"):
        Nw = g["ref_words"].sum()
        wer = (g["sub"] + g["del"] + g["ins"]).sum() / Nw
        df = pd.concat([df, pd.DataFrame([{"language": "ALL", "asr": asr,
            "n_utts": g["n_utts"].sum(), "ref_words": Nw, "wer": round(wer, 4),
            "cer": round((g["cer"] * g["ref_words"]).sum() / Nw, 4),
            "sub": g["sub"].sum(), "del": g["del"].sum(), "ins": g["ins"].sum()}])],
            ignore_index=True)

    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({'diacritics folded' if args.strip_diacritics else 'diacritics kept'})\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
