from transformers import pipeline
import huggingface_hub
import re
def transcribe_medasr(df, language):
    if language != "english":
         raise ValueError(f"Language {language} not supported by MedASR")
    model_id = "google/medasr"

    pipe = pipeline("automatic-speech-recognition", model=model_id)
    for index, row in df.iterrows():
        audio_path = row["audio_path"]
        result = pipe(audio_path)["text"]
        # remove newlines from result, anything between [], <>, and () and extra spaces
        result = result.replace("\n", " ")
        result = re.sub(r"\[.*?\]", "", result)
        result = re.sub(r"<.*?>", "", result)
        result = re.sub(r"\(.*?\)", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        df.at[index, "hypothesis"] = result
    # the chunk length is how long in seconds MedASR batches audio and the stride length is the overlap between chunks.
    return df