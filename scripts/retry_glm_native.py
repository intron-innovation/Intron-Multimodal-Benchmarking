"""Retry the rate-limited (ERROR 429) glm-5.2 rows in the native dirs, SERIALLY with
per-row CSV writes (steady throughput under Zhipu's rate limit, resilient to restarts).
Uses the native prompt. Idempotent: only touches rows still marked ERROR."""
import glob
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q

KEY = q.get_key("zhipu")
BASE = q.PROVIDERS["zhipu"]["base_url"]


def call(prompt):
    for i in range(6):
        try:
            out = q.call_openai("glm-5.2", prompt, KEY, BASE)
            if not str(out).startswith("ERROR"):
                return out
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" not in msg and "rate" not in msg.lower() and "1302" not in msg:
                return f"ERROR: {type(e).__name__}: {msg[:150]}"
            time.sleep(8 + 6 * i + random.random() * 4)  # only back off on rate limit
    return "ERROR: rate-limited after retries"


for dr in ["results/text_qa_native", "results/text_qa_omni_native"]:
    for f in sorted(glob.glob(os.path.join(q.REPO_ROOT, dr, "glm-5.2_*.csv"))):
        d = pd.read_csv(f)
        idx = [i for i in d.index if str(d.at[i, "model_answer"]).startswith("ERROR")]
        for n, i in enumerate(idx, 1):
            p = q.NATIVE_PROMPT.format(language=str(d.at[i, "language"]).capitalize(),
                                       question=str(d.at[i, "question"]))
            d.at[i, "model_answer"] = call(p)
            d.to_csv(f, index=False)          # persist after each row
            time.sleep(1.0)                    # gentle pacing to stay under the limit
        if idx:
            rem = int(d["model_answer"].astype(str).str.startswith("ERROR").sum())
            print(f"{os.path.basename(f)} ({os.path.basename(dr)}): retried {len(idx)}, remaining {rem}", flush=True)
print("done", flush=True)
