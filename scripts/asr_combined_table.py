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


def main():
    raw = pd.read_csv(os.path.join(BENCH, "asr_wer_by_language.csv")).set_index(["language", "asr"])
    fold = pd.read_csv(os.path.join(BENCH, "asr_wer_diacfold.csv")).set_index(["language", "asr"])
    diac = diac_counts()
    rows = []
    for lang in ORDER:
        for asr in ASR_ORDER:
            if (lang, asr) not in raw.index:
                continue
            rr, ff = raw.loc[(lang, asr)], fold.loc[(lang, asr)]
            has = diac.get(lang, 0) > 50
            rows.append({
                "language": lang, "asr": asr, "n": int(rr["n_utts"]),
                "WER_raw": f"{rr['wer']*100:.2f}%",
                "WER_folded": f"{ff['wer']*100:.2f}%" if has else "—",
                "CER_raw": f"{rr['cer']*100:.2f}%",
                "CER_folded": f"{ff['cer']*100:.2f}%" if has else "—",
                "diacritics_in_ref": diac.get(lang, ""),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BENCH, "asr_wer_combined.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
