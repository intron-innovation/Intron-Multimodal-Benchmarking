import pandas as pd
import json 
import os
import argparse


def infer_gemma4(df = None):
    # import gemma4 transcription function which is in model/gemma4.py
    from models.gemma4 import transcribe_gemma
    results = []
    df['reference'] = df['text']
    df['hypothesis'] = ""
    
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
        results.append(transcribe_gemma(row['audio_path']))
    df["hypothesis"] = [res['content'] for res in results]
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/transcription/gemma4_{lang}.csv", index=False)
    

def infer_intron_local(df = None):
    from models.intron_sahara import intron_local_transcribe
    df["hypothesis"] = ""
    df["reference"] = df["text"]
    for lang, group in df.groupby("language"):
        transcribed_group = intron_local_transcribe(group, lang)
        transcribed_group.to_csv(f"results/transcription/sahara_{lang}.csv", index=False)

def infer_medasr(df = None):
    from models.medasr import transcribe_medasr
    df["hypothesis"] = ""
    df["reference"] = df["text"]
    df = df[df["language"] == "english"]
    transcribed_df = transcribe_medasr(df, "english")
    transcribed_df.to_csv(f"results/transcription/medasr_english.csv", index=False)

def infer_omni_ctc(df = None):
    from models.meta_omniasr import omni_transcribe_ctc
    df["hypothesis"] = ""
    df["reference"] = df["text"]
    for lang, group in df.groupby("language"):
        transcribed_group = omni_transcribe_ctc(group, lang)
        transcribed_group.to_csv(f"results/transcription/omnictc_{lang}.csv", index=False)

def infer_omni_llm(df = None):
    from  models.meta_omniasr import omni_transcribe_llm
    df["hypothesis"] = ""
    for lang, group in df.groupby("language"):
        transcribed_group = omni_transcribe_llm(group, lang)
        transcribed_group.to_csv(f"results/transcription/omnillm_{lang}.csv", index=False)

def main():
    # parse command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()
    print(args.model)
    df = pd.read_csv("data/Transcription/meta_data.csv")
    # sample 5 for each langauge

    

    # check if files exist
    #get current working directory and prepend to audio path
    cwd = os.getcwd()
    
    
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(cwd,'data', x))
    
    df["file_exists"] = df["audio_path"].apply(lambda x: os.path.exists(x))
    # sample 5 for each langauge

    df = df[df["file_exists"]]
    
    sample_df = df.groupby("language").apply(lambda x: x.sample(3, random_state=42)).reset_index(drop=True)
    
    # run inference
    if args.model == "gemma4":
        infer_gemma4(sample_df)
    elif args.model == "sahara":
        infer_intron_local(sample_df)
    elif args.model == "medasr":
        infer_medasr(sample_df)
    elif args.model == "omnilingual_ctc":
        infer_omni_ctc(sample_df)
    elif args.model == "omnilingual_llm":
        infer_omni_llm(sample_df)
    else:
        raise ValueError("Model not supported")








if __name__ == "__main__":
    main()