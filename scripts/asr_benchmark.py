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

import jiwer
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAHARA_DIR = os.path.join(REPO, "outputs", "gates_transcription")
OMNI_ALL = os.path.join(REPO, "outputs", "gates_transcription_omni", "omni_all.csv")

# jiwer tokenization: collapse ALL whitespace to single spaces (matches str.split), then
# reduce to words (WER) / characters incl. spaces (CER). Content normalization is done by
# normalize() beforehand; these transforms only tokenize.
_WT = jiwer.Compose([jiwer.SubstituteRegexes({r"\s+": " "}), jiwer.Strip(),
                     jiwer.ReduceToListOfListOfWords()])
_CT = jiwer.Compose([jiwer.SubstituteRegexes({r"\s+": " "}), jiwer.Strip(),
                     jiwer.ReduceToListOfListOfChars()])
# 'literal' mode: NO whitespace normalization at all — split on single space (keeps empty
# tokens from double spaces; newlines stay attached), chars counted incl. every space/newline.
_WT_LIT = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
_CT_LIT = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])


def normalize(s, mode="keep"):
    """mode: 'none' = literal (case-sensitive, punctuation kept, whitespace-tokenized);
    'keep' = lowercase + strip punctuation + collapse whitespace (diacritics kept);
    'fold' = keep + also strip diacritics/combining marks."""
    s = "" if pd.isna(s) else str(s)
    if mode in ("literal", "default"):
        return s                            # fully raw; 'default' lets jiwer's own default transform run
    if mode == "none":
        return s.strip()                    # only tokenization (via .split) normalizes whitespace
    s = s.lower().strip()
    if mode == "fold":
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    # extra ASRs kept in their own dirs: intron voice API, gemini (any subset present)
    for asr, sub_dir, prefix in [("intron", "gates_transcription_intron", "intron_"),
                                 ("gemini", "gates_transcription_gemini", "gemini_")]:
        for f in sorted(glob.glob(os.path.join(REPO, "outputs", sub_dir, f"{prefix}*.csv"))):
            sub = os.path.basename(f).replace(prefix, "").replace(".csv", "")
            d = pd.read_csv(f)
            d["audio_id"] = d["audio_id"].astype(str)
            hyps[(asr, sub)] = dict(zip(d["audio_id"], d["hypothesis"]))
    return refs, hyps


def score(refs, hyps, mode):
    rows = []
    for (asr, sub), hyp_map in sorted(hyps.items()):
        ref_map = refs.get(sub, {})
        R, H = [], []
        for a in hyp_map:
            if a not in ref_map:
                continue
            r = normalize(ref_map[a], mode)
            if not r.strip():                      # jiwer errors on empty references
                continue
            R.append(r); H.append(normalize(hyp_map[a], mode))
        if not R:
            continue
        if mode == "default":                  # plain jiwer.wer/cer — jiwer's own default transforms
            wo = jiwer.process_words(R, H)
            co = jiwer.process_characters(R, H)
        else:
            wt, ct = (_WT_LIT, _CT_LIT) if mode == "literal" else (_WT, _CT)
            wo = jiwer.process_words(R, H, reference_transform=wt, hypothesis_transform=wt)
            co = jiwer.process_characters(R, H, reference_transform=ct, hypothesis_transform=ct)
        S, D, I = wo.substitutions, wo.deletions, wo.insertions
        Nw = S + D + wo.hits                        # reference words = hits + subs + dels
        rows.append({"language": sub, "asr": asr, "n_utts": len(R),
                     "ref_words": Nw, "wer": round(wo.wer, 4), "cer": round(co.cer, 4),
                     "sub": S, "del": D, "ins": I})
    return pd.DataFrame(rows).sort_values(["language", "asr"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip-diacritics", action="store_true",
                    help="fold combining marks before scoring (same as --norm fold)")
    ap.add_argument("--norm", choices=["default", "literal", "none", "keep", "fold"], default=None,
                    help="default=plain jiwer.wer (jiwer's own default transform, no extra norm); "
                         "literal=zero normalization incl. whitespace (harshest); "
                         "none=raw text but whitespace-tokenized; "
                         "keep=lowercase+strip punct (diacritics kept); fold=strip diacritics too")
    ap.add_argument("--out", default="results/asr_benchmark/asr_wer_by_language.csv")
    args = ap.parse_args()

    mode = args.norm or ("fold" if args.strip_diacritics else "keep")
    refs, hyps = load_hyps(mode)
    df = score(refs, hyps, mode)

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
