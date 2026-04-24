import pandas as pd
import json 
import os
import argparse




def infer_gemma4(df = None):
    # import gemma4 translation function which is in model/gemma4.py
    from models.gemma4 import translate_gemma
    results = []
    df['reference'] = df['translation']
    df['hypothesis'] = ""
    
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
        results.append(translate_gemma(row['audio_path'], row['language'], 'en'))
    df["hypothesis"] = [res['content'] for res in results]
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/translation/gemma4_{lang}.csv", index=False)
    
def infer_gemini(df = None, model_name = "gemini-3-flash-preview"):
    # import gemini 3 
    from models.proprietary_models import gemini_translate
    results = []
    df['reference'] = df['translation']
    df['hypothesis'] = ""
    df = pd.read_csv("/mnt/multilingual_data/audio/Intron-Multimodal-Benchmarking/results/translation/gemini_translate_rerun.csv")
    #filter_out rows where hypothesis is not ERROR
    df1 = df[df['hypothesis'] != "ERROR"]
    df2 = df[df['hypothesis'] == "ERROR"]
    for index, row in df2.iterrows():
        print(f"Processing {index+1}/{len(df2)}: {row['audio_path']}")
        results.append(gemini_translate(row['audio_path'], row['language'], 'en', model_name))
    df2["hypothesis"] = results
    df = pd.concat([df1, df2])
    # for index, row in df.iterrows():
    #     print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
    #     results.append(gemini_translate(row['audio_path'], row['language'], 'en', model_name))
    # df["hypothesis"] = results
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/translation/{model_name}_{lang}.csv", index=False)     
def infer_gpt_4o_translate(df = None, model_name = "gpt-4o-audio-preview"):
    from models.proprietary_models import gpt4o_audio_translate
    results = []
    df['reference'] = df['translation']
    df['hypothesis'] = ""
    # df = pd.read_csv("/mnt/multilingual_data/audio/Intron-Multimodal-Benchmarking/results/translation/gpt4o_audio_translate_rerun.csv")
    # #filter_out rows where hypothesis is not ERROR
    # df1 = df[df['hypothesis'] != "ERROR"]
    # df2 = df[df['hypothesis'] == "ERROR"]
    # for index, row in df2.iterrows():
    #     print(f"Processing {index+1}/{len(df2)}: {row['audio_path']}")
    #     results.append(gpt4o_audio_translate(row['audio_path'], row['language'], 'en'))
    # df2["hypothesis"] = results
    # df = pd.concat([df1, df2])
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
        results.append(gpt4o_audio_translate(row['audio_path'], row['language'], 'en'))
    df["hypothesis"] = results
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/translation/{model_name}_{lang}.csv", index=False)
def infer_qwen3_translate(df = None, model_name = "qwen3-livetranslate-flash"):
    from models.proprietary_models import qwen3_translate
    results = []
    df['reference'] = df['translation']
    df['hypothesis'] = ""
    # df = pd.read_csv("/mnt/multilingual_data/audio/Intron-Multimodal-Benchmarking/results/translation/qwen3_translate_rerun.csv")
    # #filter_out rows where hypothesis is not ERROR
    # df1 = df[df['hypothesis'] != "ERROR"]
    # df2 = df[df['hypothesis'] == "ERROR"]
    # for index, row in df2.iterrows():
    #     print(f"Processing {index+1}/{len(df2)}: {row['audio_path']}")
    #     results.append(qwen3_audio_translate(row['audio_path'], row['language'], 'en'))
    # df2["hypothesis"] = results
    # df = pd.concat([df1, df2])
    df = df[df['language'] == "french"] # qwen3 does not support english to english translation, so we filter out english samples
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
        results.append(qwen3_translate(row['audio_path'], row['language'], 'en'))
    df["hypothesis"] = results
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/translation/{model_name}_{lang}.csv", index=False)

def main():
    # parse command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()
    print(args.model)
    df = pd.read_csv("data/Translation/meta_data.csv")
   
    

    # check if files exist
    #get current working directory and prepend to audio path
    cwd = os.getcwd()
    
    
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(cwd, x))
    
    df["file_exists"] = df["audio_path"].apply(lambda x: os.path.exists(x))

    print(df["file_exists"].value_counts())
  
    df = df[df["file_exists"]]

    if args.model == "gemma4":
        infer_gemma4(df)
    elif args.model == "gemini_3_flash":
        infer_gemini(df, model_name = "gemini-3-flash-preview")
    elif args.model == "gpt4o_audio_translate":
        infer_gpt_4o_translate(df, model_name = "gpt-4o-audio-preview")
    elif args.model == "qwen3_translate":
        infer_qwen3_translate(df, model_name = "qwen3-livetranslate-flash")
    else:
        raise ValueError("Model not supported")






if __name__ == "__main__":
    main()