from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torchaudio
import os
import pandas as pd
lang_map = {
    "afrikaans": "afr_Latn",
    "akan": "aka_Latn",
    "amharic": "amh_Ethi",
    "arabic": "arb_Arab",
    "english": "eng_Latn",
    "french": "fra_Latn",
    "hausa": "hau_Latn",
    "igbo": "ibo_Latn",
    "kinyarwanda": "kin_Latn",
    "pedi": "nso_Latn",
    "sesotho": "nso_Latn", 
    "shona": "sna_Latn",
    "swahili": "swh_Latn",
    "tswana": "tsn_Latn",
    "xhosa": "xho_Latn",
    "yoruba": "yor_Latn",
    "zulu": "zul_Latn"
}
def transcribe_in_chunks(audio_path, out_dir, asr_model, chunk_len=30.0, overlap=0.0, sr=16000,model_type="nemo", language=None):
    """
    Splits audio into fixed-length chunks and runs ASR model on each chunk.
    
    Args:
        audio_path (str): Path to input audio file
        out_dir (str): Directory for temporary chunk files
        asr_model: ASR model with a .transcribe() method
        chunk_len (float): Chunk length in seconds
        overlap (float): Overlap between chunks in seconds
        sr (int): Target sample rate
        language (str): Language code for the ASR model
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
            text = asr_model.transcribe([out_path])[0]  # adjust if your model returns differently
            transcripts.append(text.text.strip())
        else:
            text = asr_model.transcribe([out_path], lang=[language])[0]  # adjust if your model returns differently
            transcripts.append(text.strip())

        i += 1
        start += step

    # Join into one full transcript
    full_transcript = " ".join(transcripts)
    return full_transcript





def omni_transcribe(df, language, pipeline):
    language = lang_map[language]
    test_df_1 = df[df['duration']<40]
    test_df_2 = df[df['duration']>=40]
    
    
    transcriptions = pipeline.transcribe(test_df_1["audio_path"].tolist(), lang=[language for _ in range(len(test_df_1))], batch_size=8)
    test_df_1['hypothesis'] = transcriptions
    test_df_2['hypothesis'] = test_df_2['audio_path'].apply(lambda x: transcribe_in_chunks(x, out_dir="/tmp/chunks", asr_model=pipeline, chunk_len=39.0, overlap=0.2, sr=16000, model_type="omni" , language=language))

    return pd.concat([test_df_1, test_df_2], ignore_index=True)



def load_omni_model_pipeline(model_name):
    if model_name == "omniASR_CTC_7B_v2":
        pipeline = ASRInferencePipeline(model_card="omniASR_CTC_7B_v2", device="cuda")
    elif model_name == "omniASR_LLM_7B_v2":
        pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_v2", device="cuda")
    elif model_name == "omniASR_LLM_300M_v2":
        pipeline = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2", device="cuda")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return pipeline