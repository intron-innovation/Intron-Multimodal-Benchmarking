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



def infer_omni(df = None , model_name = "omniASR_LLM_7B_v2"):
    from  models.meta_omniasr import omni_transcribe
    from models.meta_omniasr import load_omni_model_pipeline
    df["hypothesis"] = ""
    pipeline = load_omni_model_pipeline(model_name)
    for lang, group in df.groupby("language"):
        print(f"Processing language: {lang}")
        transcribed_group = omni_transcribe(group, lang, pipeline)
        if model_name == "omniASR_CTC_7B_v2":
            transcribed_group.to_csv(f"results/transcription/omniCTC_{lang}.csv", index=False)
        else:
            transcribed_group.to_csv(f"results/transcription/omnillm_{lang}.csv", index=False)

def infer_qwen3(df = None , model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"):
    from  models.qwen3_omni import transcribe_qwen3omni
    from models.qwen3_omni import load_model
    
    df["hypothesis"] = ""
    
    model, processor = load_model(model_name)
    # include only english and french
    df['language'] = df['language'].str.lower()
    df = df[df["language"].isin(["english", "french"])]
    for lang, group in df.groupby("language"):
        print(f"Processing language: {lang}")
        for index, row in group.iterrows():
            print(f"Processing {index+1}/{len(group)}: {row['audio_path']}")
            text = transcribe_qwen3omni(row['audio_path'], model, processor)
            group.at[index, "hypothesis"] = text
        group.to_csv(f"results/transcription/qwen3-omni-instruct_{lang}.csv", index=False)
        

def main():
    # parse command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()
    print(args.model)
    df = pd.read_csv("data/Transcription/meta_data.csv")
  

    

    # check if files exist
    #get current working directory and prepend to audio path
    cwd = os.getcwd()
    
    
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(cwd,'data', x))
    
    df["file_exists"] = df["audio_path"].apply(lambda x: os.path.exists(x))
   
    df = df[df["file_exists"]]
    
    # sample_df = df.groupby("language").apply(lambda x: x.sample(3, random_state=42)).reset_index(drop=True)
    sample_df = df 
    # run inference
    if args.model == "gemma4":
        infer_gemma4(sample_df)
    elif args.model == "sahara":
        infer_intron_local(sample_df)
    elif args.model == "medasr":
        infer_medasr(sample_df)
    elif args.model == "omnilingual_ctc":
        infer_omni(sample_df, model_name="omniASR_CTC_7B_v2")
    elif args.model == "omnilingual_llm":
        infer_omni(sample_df, model_name="omniASR_LLM_7B_v2")
    elif args.model == "qwen3_omni":
        infer_qwen3(sample_df, model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    elif args.model == "qwen3_omni_thinking":
        infer_qwen3(sample_df, model_name="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    else:
        raise ValueError("Model not supported")








if __name__ == "__main__":
    main()