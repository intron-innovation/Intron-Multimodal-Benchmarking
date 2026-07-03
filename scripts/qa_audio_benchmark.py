"""
Audio-input clinical QA benchmark: send the spoken question (WAV) directly to
audio-capable models and collect an English answer. Complements qa_text_benchmark.py
(which feeds the transcribed text). Input audio + language come from the transcription
outputs (outputs/gates_transcription/*.csv -> local_path). Outputs go to
results/audio_qa/<label>_<subset>.csv.

Adapters:
  google_audio    - genai file upload + generate_content([audio, prompt])
  ollama_audio    - OpenAI-compatible chat with input_audio (local gemma4:e4b)
  openai_realtime - Realtime API (WebSocket): audio in -> text out (gpt-realtime-2)

Run (voicebot venv):
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/qa_audio_benchmark.py --list
  /data/avx-voice-bot-engine/voicebot/venv/bin/python scripts/qa_audio_benchmark.py --models gemini-flash,gemini-3.1-pro
"""
import argparse
import base64
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_text_benchmark as q  # reuse env loading + get_key + REPO paths

REPO_ROOT = q.REPO_ROOT
IN_DIR = q.IN_DIR  # outputs/gates_transcription
OUT_DIR = os.path.join(REPO_ROOT, "results", "audio_qa")

AUDIO_PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a correct and concise "
    "answer in English based on your clinical knowledge and experience, explaining the "
    "reasoning behind your response using medical evidence and current locally relevant "
    "practices to support your answer.  The question is spoken in the provided audio, in "
    "{language} language, spoken in Nigeria. Listen to the audio and answer the question. "
    "Provide the answer ONLY in English language. No other language should be present in "
    "your answer. Be concise, presenting your answer in 5 sentences or less."
)

# Same prompt but answers in the question's own language (not forced to English).
NATIVE_AUDIO_PROMPT = (
    "You are an expert medical doctor with years of experience in Africa, working in a "
    "community-based hospital setting. Your expertise covers a wide range of medical "
    "specialties and conditions from common ailments to complex diseases. You are tasked "
    "with answering health-related questions. You should provide a correct and concise "
    "answer based on your clinical knowledge and experience, explaining the reasoning "
    "behind your response using medical evidence and current locally relevant practices to "
    "support your answer. The question is spoken in the provided audio, in {language} "
    "language, spoken in Nigeria. Listen to the audio and answer the question in {language} "
    "language, the same language the question is asked in. Be concise, presenting your "
    "answer in 5 sentences or less."
)

REGISTRY = [
    {"requested": "Gemini 3.5 flash (audio)", "label": "gemini-flash",   "kind": "google_audio",    "key": "google", "model_id": "gemini-flash-latest"},
    {"requested": "Gemini 3.1 Pro (audio)",   "label": "gemini-3.1-pro", "kind": "google_audio",    "key": "google", "model_id": "gemini-3.1-pro-preview"},
    {"requested": "Gemma4-E4B (audio)",       "label": "gemma4-e4b",     "kind": "ollama_audio",    "key": "ollama", "model_id": "gemma4:e4b"},
    {"requested": "GPT Realtime 2 (audio)",   "label": "gpt-realtime-2", "kind": "openai_realtime", "key": "openai", "model_id": "gpt-realtime-2"},
]


# ---- adapters ---------------------------------------------------------------
def google_audio(model_id, audio_path, prompt, key):
    from google import genai
    from google.genai import types

    # timeout (ms) so a hung upload/generate raises instead of blocking forever
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))
    f = client.files.upload(file=audio_path)
    return (client.models.generate_content(model=model_id, contents=[f, prompt]).text or "").strip()


def ollama_audio(model_id, audio_path, prompt, key):
    from openai import OpenAI

    b64 = base64.b64encode(open(audio_path, "rb").read()).decode()
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=300)
    r = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]}],
    )
    return (r.choices[0].message.content or "").strip()


def _pcm16_24k_base64(audio_path):
    """Realtime wants raw mono 24kHz PCM16, base64-encoded."""
    import torchaudio

    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    pcm16 = (wav.clamp(-1, 1) * 32767).to("cpu").short().numpy().tobytes()
    return base64.b64encode(pcm16).decode()


def openai_realtime(model_id, audio_path, prompt, key):
    from openai import OpenAI

    client = OpenAI(api_key=key)
    audio_b64 = _pcm16_24k_base64(audio_path)
    text_parts = []
    with client.realtime.connect(model=model_id) as conn:
        conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": prompt,
            # disable server VAD so our manual commit + response.create drives the turn
            "audio": {"input": {"turn_detection": None}},
        })
        conn.input_audio_buffer.append(audio=audio_b64)
        conn.input_audio_buffer.commit()
        conn.response.create()
        for event in conn:
            t = getattr(event, "type", "")
            if t.endswith("output_text.delta") or t.endswith("text.delta"):
                text_parts.append(event.delta)
            elif t == "response.done" or t.endswith("response.completed"):
                break
            elif t == "error":
                return f"ERROR: realtime {getattr(event, 'error', '')}"
    return "".join(text_parts).strip()


ADAPTERS = {
    "google_audio": google_audio,
    "ollama_audio": ollama_audio,
    "openai_realtime": openai_realtime,
}


def is_reachable(m):
    return bool(q.get_key(m["key"]))


def answer_one(m, audio_path, prompt):
    try:
        return ADAPTERS[m["kind"]](m["model_id"], audio_path, prompt, q.get_key(m["key"]))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"


def load_audio_index(languages=None, limit=None):
    frames = []
    for f in sorted(glob.glob(os.path.join(IN_DIR, "sahara_*.csv"))):
        d = pd.read_csv(f)
        d["subset"] = os.path.basename(f).replace("sahara_", "").replace(".csv", "")
        if limit:
            d = d.head(limit)
        frames.append(d[["audio_id", "language", "subset", "local_path", "reference", "url"]])
    df = pd.concat(frames, ignore_index=True)
    if languages:
        df = df[df["language"].str.lower().isin(languages)]
    return df


def run_model(m, df, force=False, workers=4, out_dir=OUT_DIR, prompt_template=AUDIO_PROMPT):
    for subset, group in df.groupby("subset"):
        lang = group["language"].iloc[0]
        out_path = os.path.join(out_dir, f"{m['label']}_{subset}.csv")
        if os.path.exists(out_path) and not force:
            print(f"  skip (exists): {out_path}")
            continue
        group = group.reset_index(drop=True)
        prompt = prompt_template.format(language=str(lang).capitalize())
        paths = list(group["local_path"])

        def work(idx):
            p = paths[idx]
            return answer_one(m, p, prompt) if isinstance(p, str) and os.path.exists(p) else "ERROR: missing audio"

        with ThreadPoolExecutor(max_workers=workers) as ex:
            answers = list(ex.map(work, range(len(paths))))

        out = group.copy()
        out["model_answer"] = answers
        out["requested_model"] = m["requested"]
        out["model_id"] = m["model_id"]
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--languages", type=str, default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--native", action="store_true", help="answer in the question's own language")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in REGISTRY:
            print(f"{m['label']:16s} {m['kind']:16s} {m['model_id']:22s} "
                  f"{'YES' if is_reachable(m) else 'no'}")
        return

    want = {x.strip() for x in args.models.split(",") if x.strip()}
    langs = {x.strip().lower() for x in args.languages.split(",") if x.strip()} or None
    df = load_audio_index(languages=langs, limit=args.limit)
    print(f"{len(df)} audio questions loaded")

    for m in REGISTRY:
        if want and m["label"] not in want:
            continue
        if not want and not is_reachable(m):
            print(f"skip (unreachable): {m['label']}")
            continue
        print(f"\n=== {m['label']}  ({m['requested']} -> {m['model_id']})")
        run_model(m, df, force=args.force, workers=args.workers, out_dir=args.out_dir,
                  prompt_template=(NATIVE_AUDIO_PROMPT if args.native else AUDIO_PROMPT))


if __name__ == "__main__":
    main()
