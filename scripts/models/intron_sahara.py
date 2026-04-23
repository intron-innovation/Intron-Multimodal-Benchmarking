import os
import pandas as pd
# load the .env file
from dotenv import load_dotenv
load_dotenv()

import torch
import nemo.collections.asr as nemo_asr

import torchaudio
import os

def transcribe_in_chunks(audio_path, out_dir, asr_model, chunk_len=30.0, overlap=0.0, sr=16000,model_type="nemo"):
    """
    Splits audio into fixed-length chunks and runs ASR model on each chunk.
    
    Args:
        audio_path (str): Path to input audio file
        out_dir (str): Directory for temporary chunk files
        asr_model: ASR model with a .transcribe() method
        chunk_len (float): Chunk length in seconds
        overlap (float): Overlap between chunks in seconds
        sr (int): Target sample rate
    
    Returns:
        str: Concatenated transcription of all chunks
        list: List of per-chunk transcripts
    """
    os.makedirs(out_dir, exist_ok=True)
    waveform, orig_sr = torchaudio.load(audio_path)

    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    total_len_sec = waveform.size(1) / sr
    step = chunk_len - overlap
    transcripts = []

    start = 0.0
    i = 0
    while start < total_len_sec:
        end = min(start + chunk_len, total_len_sec)
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        chunk = waveform[:, start_sample:end_sample]

        out_path = os.path.join(out_dir, f"{os.path.basename(audio_path).split('.')[0]}_chunk_{i}.wav")
        torchaudio.save(out_path, chunk, sr)

        # Run ASR model
        # average the channels if stereo
        if chunk.size(0) > 1:
            chunk = chunk.mean(dim=0, keepdim=True)
        if model_type == "nemo":
            text = asr_model.transcribe([out_path], channel_selector=0)[0]  # adjust if your model returns differently
            transcripts.append(text.text.strip())
        else:
            text = asr_model.transcribe([out_path])[0]  # adjust if your model returns differently
            transcripts.append(text.strip())

        i += 1
        start += step

    # Join into one full transcript
    full_transcript = " ".join(transcripts)
    return full_transcript

def intron_local_transcribe(df, language):
    local_dir = os.environ.get("INTRON_LOCAL_DIR", "NOT_SET")
    model_path = os.path.join(local_dir, f"{language}.nemo")
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found for language {language} at path {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    # split 45 seconds audios and <= 45 seconds audios
    df_short = df[df["duration"] <= 45]
    df_long = df[df["duration"] > 45]
    df_short_hyp = model.transcribe(df_short["audio_path"].tolist(), batch_size=16, num_workers=4,channel_selector=0)
    df_short["hypothesis"] = [hyp.text for hyp in df_short_hyp]

    df_long["hypothesis"] = df_long["audio_path"].apply(lambda x: transcribe_in_chunks(x, "temp_chunks", model, model_type="nemo"))
    return pd.concat([df_short, df_long])

