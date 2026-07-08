"""
Build the consolidated ASR table (results/asr_benchmark/asr_wer_combined.csv) from the
raw + diacritics-folded benchmark CSVs, for every ASR present (sahara, omni, intron, ...).
Folded columns are shown only for languages whose reference carries diacritics.
"""
import glob
import os
import unicodedata

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "results", "asr_benchmark")
ORDER = ["good_english", "accented_english", "pidgin", "hausa", "yoruba", "fulfulde", "ALL"]
ASR_ORDER = ["sahara", "omni", "intron", "gemini"]


def diac_counts():
    out = {}
    for f in glob.glob(os.path.join(REPO, "outputs", "gates_transcription", "sahara_*.csv")):
        s = os.path.basename(f).replace("sahara_", "").replace(".csv", "")
        d = pd.read_csv(f)
        out[s] = sum(1 for x in d["reference"] for ch in unicodedata.normalize("NFD", str(x))
                     if unicodedata.category(ch) == "Mn")
    return out


def _load(name):
    p = os.path.join(BENCH, name)
    return pd.read_csv(p).set_index(["language", "asr"]) if os.path.exists(p) else None


def main():
    # three normalization levels: unnormalized (literal), normalized (diacritics kept), folded
    unn = _load("asr_wer_unnormalized.csv")
    norm = _load("asr_wer_by_language.csv")
    fold = _load("asr_wer_diacfold.csv")
    diac = diac_counts()
    rows = []
    for lang in ORDER:
        for asr in ASR_ORDER:
            if norm is None or (lang, asr) not in norm.index:
                continue
            nr, ff = norm.loc[(lang, asr)], fold.loc[(lang, asr)]
            un = unn.loc[(lang, asr)] if (unn is not None and (lang, asr) in unn.index) else None
            has = diac.get(lang, 0) > 50
            pct = lambda v: f"{v*100:.2f}%"
            rows.append({
                "language": lang, "asr": asr, "n": int(nr["n_utts"]),
                "WER_unnormalized": pct(un["wer"]) if un is not None else "—",
                "WER_normalized": pct(nr["wer"]),
                "WER_folded": pct(ff["wer"]) if has else "—",
                "CER_unnormalized": pct(un["cer"]) if un is not None else "—",
                "CER_normalized": pct(nr["cer"]),
                "CER_folded": pct(ff["cer"]) if has else "—",
                "diacritics_in_ref": diac.get(lang, ""),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BENCH, "asr_wer_combined.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
