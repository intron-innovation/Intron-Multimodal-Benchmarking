"""
Text-based clinical QA benchmark over the transcribed gates_data questions.

Feeds each transcribed question (the Sahara STT `hypothesis`) to a set of LLMs and
collects an English answer. Input comes from outputs/gates_transcription/*.csv;
outputs go to results/text_qa/<label>_<language>.csv.

Providers: Google (genai) and Anthropic (Messages API) have bespoke adapters;
everything else (OpenAI, DashScope/Qwen, DeepSeek, Zhipu/GLM, OSS hosts) speaks the
OpenAI chat-completions protocol, so a new provider is just a base_url + key.

Models whose API key is missing (or whose model_id is unknown) are skipped cleanly,
so this can run today with only Google+OpenAI keys and fill in the rest later.

Run (voicebot venv has the SDKs):
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/qa_text_benchmark.py --list
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/qa_text_benchmark.py --limit 2
"""
import argparse
import glob
import os
import re

import pandas as pd
from dotenv import dotenv_values

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(REPO_ROOT, "outputs", "gates_transcription")
OUT_DIR = os.path.join(REPO_ROOT, "results", "text_qa")
ENV_FILES = [
    os.path.join(REPO_ROOT, ".env"),
    "/home/azureuser/llm_agents_benchmark/.env",
]

# Substrings (normalized: lowercase, alphanumerics only) that identify each
# provider's key in a possibly messily-named .env (e.g. "openAI-api_key", "glm_api").
KEY_PATTERNS = {
    "google": ["gemini", "genai", "googleapi"],
    "openai": ["openai"],
    "anthropic": ["anthropic", "claude"],
    "dashscope": ["dashscope", "qwen"],
    "deepseek": ["deepseek"],
    "zhipu": ["zhipu", "glm", "bigmodel"],
    "oss": ["ossapi", "groq", "together", "fireworks"],
    "microsoft": ["maithinking", "maiapi"],
}

# Provider config: how to reach it, and which env var holds the key.
PROVIDERS = {
    "google":    {"kind": "google",    "key": "GEMINI_API_KEY",    "base_url": None},
    "openai":    {"kind": "openai",     "key": "OPENAI_API_KEY",    "base_url": None},
    "anthropic": {"kind": "anthropic",  "key": "ANTHROPIC_API_KEY", "base_url": None},
    "dashscope": {"kind": "openai",     "key": "DASHSCOPE_API_KEY",
                  "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    "deepseek":  {"kind": "openai",     "key": "DEEPSEEK_API_KEY",  "base_url": "https://api.deepseek.com"},
    "zhipu":     {"kind": "openai",     "key": "ZHIPU_API_KEY",
                  "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "oss":       {"kind": "openai",     "key": "OSS_API_KEY",       "base_url": os.environ.get("OSS_BASE_URL")},
    "microsoft": {"kind": "openai",     "key": "MAI_API_KEY",       "base_url": os.environ.get("MAI_BASE_URL")},
    "ollama":    {"kind": "openai",     "key": None,                "base_url": "http://localhost:11434/v1"},
}

# Requested model -> (label, provider, model_id). model_id=None means "unknown / confirm".
REGISTRY = [
    {"requested": "Gemini 3.5 flash (audio)", "label": "gemini-flash",     "provider": "google",    "model_id": "gemini-flash-latest"},
    {"requested": "Gemini 3.1 Pro (audio)",   "label": "gemini-3.1-pro",   "provider": "google",    "model_id": "gemini-3.1-pro-preview"},
    {"requested": "GPT 5.5",                  "label": "gpt-5.5",          "provider": "openai",    "model_id": "gpt-5.5"},
    {"requested": "GPT Realtime 2 (audio)",   "label": "gpt-realtime-2",   "provider": "openai",    "model_id": "gpt-realtime-2", "audio_only": True},
    {"requested": "GPT-OSS-120B",             "label": "gpt-oss-120b",     "provider": "oss",       "model_id": "gpt-oss-120b"},
    {"requested": "Claude Opus-4.8",          "label": "claude-opus-4.8",  "provider": "anthropic", "model_id": "claude-opus-4-8"},
    {"requested": "Qwen-3.7-Plus (audio)",    "label": "qwen-3.7-plus",    "provider": "dashscope", "model_id": "qwen3.7-plus"},
    {"requested": "Qwen-3.7-Max",             "label": "qwen-3.7-max",     "provider": "dashscope", "model_id": "qwen3.7-max"},
    {"requested": "Deepseek-v4 pro",          "label": "deepseek-v4-pro",  "provider": "deepseek",  "model_id": "deepseek-v4-pro"},
    {"requested": "GLM-5.2-max",              "label": "glm-5.2",          "provider": "zhipu",     "model_id": "glm-5.2"},
    {"requested": "MAI-Thinking-1",           "label": "mai-thinking-1",   "provider": "microsoft", "model_id": "MAI-Thinking-1"},
    {"requested": "Gemma4-E4B (Ollama)",      "label": "gemma4-e4b",       "provider": "ollama",    "model_id": "gemma4:e4b"},
]

PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a correct and concise "
    "answer in English based on your clinical knowledge and experience, explaining the "
    "reasoning behind your response using medical evidence and current locally relevant "
    "practices to support your answer.  The question will be in {language} language, spoken "
    "in Nigeria. Provide the answer ONLY in English language. No other language should be "
    "present in your answer. Be concise, presenting your answer in 5 sentences or less. "
    "Here is the question: {question}"
)

# Same prompt but answers in the question's own language (not forced to English).
NATIVE_PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a correct and concise "
    "answer based on your clinical knowledge and experience, explaining the reasoning "
    "behind your response using medical evidence and current locally relevant practices to "
    "support your answer. The question will be in {language} language, spoken in Nigeria. "
    "Provide the answer in {language} language, the same language the question is asked in. "
    "Be concise, presenting your answer in 5 sentences or less. Here is the question: {question}"
)

_ENV = {}
for _f in ENV_FILES:
    if os.path.exists(_f):
        _ENV.update(dotenv_values(_f))
_ENV.update(os.environ)
# normalized name -> value (strip whitespace-y / punctuation-y key names)
_NORM_ENV = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in _ENV.items()}


def get_key(provider):
    if provider == "ollama":
        return "ollama"  # local server; api_key is ignored but must be non-empty
    for nk, v in _NORM_ENV.items():
        if v and any(pat in nk for pat in KEY_PATTERNS.get(provider, [])):
            return v
    return None


def is_reachable(m):
    # audio-only models (e.g. gpt-realtime-2) are not usable in this text pipeline
    if m.get("audio_only"):
        return False
    p = PROVIDERS[m["provider"]]
    return bool(m["model_id"]) and bool(get_key(m["provider"])) and (
        p["kind"] != "openai" or p["base_url"] is not None or m["provider"] == "openai"
    )


# ---- provider adapters (return answer text, or "ERROR: ...") -----------------
def call_google(model_id, prompt, key, base_url=None):
    from google import genai

    client = genai.Client(api_key=key)
    return client.models.generate_content(model=model_id, contents=[prompt]).text


def call_openai(model_id, prompt, key, base_url=None):
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def call_anthropic(model_id, prompt, key, base_url=None):
    import requests

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model_id, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    texts = [b.get("text", "") for b in j.get("content", []) if b.get("type") == "text"]
    if not texts:
        return f"ERROR: empty content (stop_reason={j.get('stop_reason')})"
    return "".join(texts).strip()


ADAPTERS = {"google": call_google, "openai": call_openai, "anthropic": call_anthropic}


def answer_one(m, prompt):
    p = PROVIDERS[m["provider"]]
    fn = ADAPTERS[p["kind"]]
    try:
        out = fn(m["model_id"], prompt, get_key(m["provider"]), p["base_url"])
        return (out or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"


def load_questions(in_dir=IN_DIR, languages=None, limit=None, pattern="*.csv"):
    frames = []
    for f in sorted(glob.glob(os.path.join(in_dir, pattern))):
        d = pd.read_csv(f)
        # transcription column: prefer 'hypothesis', else 'text'
        if "hypothesis" in d.columns:
            d = d.rename(columns={"hypothesis": "question"})
        elif "text" in d.columns and "question" not in d.columns:
            d = d.rename(columns={"text": "question"})
        # subset: use a 'subset' column if present, else derive from filename
        if "subset" not in d.columns:
            d["subset"] = os.path.basename(f).replace("sahara_", "").replace(".csv", "")
        for c in ("audio_id", "reference", "url"):
            if c not in d.columns:
                d[c] = pd.NA
        if limit:
            d = d.groupby("subset", group_keys=False).head(limit)
        frames.append(d[["audio_id", "language", "subset", "question", "reference", "url"]])
    df = pd.concat(frames, ignore_index=True)
    if languages:
        df = df[df["language"].str.lower().isin(languages)]
    return df


def run_model(m, df, force=False, workers=8, out_dir=OUT_DIR, prompt_template=PROMPT):
    from concurrent.futures import ThreadPoolExecutor

    for subset, group in df.groupby("subset"):
        lang = group["language"].iloc[0]  # constant within a subset
        out_path = os.path.join(out_dir, f"{m['label']}_{subset}.csv")
        if os.path.exists(out_path) and not force:
            print(f"  skip (exists): {out_path}")
            continue
        group = group.reset_index(drop=True)
        qs = [str(x).strip() for x in group["question"]]
        prompts = [prompt_template.format(language=str(lang).capitalize(), question=q) for q in qs]

        def work(idx):
            return answer_one(m, prompts[idx]) if qs[idx] else ""

        with ThreadPoolExecutor(max_workers=workers) as ex:
            answers = list(ex.map(work, range(len(qs))))

        out = group.copy()
        out["model_answer"] = answers
        out["requested_model"] = m["requested"]
        out["model_id"] = m["model_id"]
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(out)} rows)")


def print_registry():
    print(f"{'requested':26s} {'label':16s} {'provider':10s} {'model_id':22s} reachable")
    for m in REGISTRY:
        print(f"{m['requested']:26s} {m['label']:16s} {m['provider']:10s} "
              f"{str(m['model_id']):22s} {'YES' if is_reachable(m) else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show model registry + reachability")
    ap.add_argument("--models", type=str, default="", help="comma list of labels to run (default: all reachable)")
    ap.add_argument("--languages", type=str, default="", help="comma list of languages")
    ap.add_argument("--limit", type=int, default=None, help="max rows per language (smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests per language")
    ap.add_argument("--in-dir", type=str, default=IN_DIR, help="dir of transcription CSVs")
    ap.add_argument("--glob", type=str, default="*.csv", help="filename glob within in-dir")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR, help="output dir for answers")
    ap.add_argument("--native", action="store_true", help="answer in the question's own language")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        print_registry()
        return

    want_labels = {x.strip() for x in args.models.split(",") if x.strip()}
    langs = {x.strip().lower() for x in args.languages.split(",") if x.strip()} or None
    df = load_questions(in_dir=args.in_dir, languages=langs, limit=args.limit, pattern=args.glob)
    print(f"{len(df)} questions loaded from {args.in_dir}")

    for m in REGISTRY:
        if want_labels and m["label"] not in want_labels:
            continue
        if not want_labels and not is_reachable(m):
            print(f"skip (unreachable): {m['label']} (missing key/model_id)")
            continue
        print(f"\n=== {m['label']}  ({m['requested']} -> {m['model_id']})")
        run_model(m, df, force=args.force, workers=args.workers, out_dir=args.out_dir,
                  prompt_template=(NATIVE_PROMPT if args.native else PROMPT))


if __name__ == "__main__":
    main()
