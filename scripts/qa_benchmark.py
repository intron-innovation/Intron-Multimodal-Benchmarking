import pandas as pd
import json 
import os
import argparse




def infer_gemma4(df = None):
    # import gemma4 translation function which is in model/gemma4.py
    from models.gemma4 import spoken_qa
    results = []
    df['reference'] = df['answer']
    df['hypothesis'] = ""
    
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: {row['audio_path']}")
        results.append(spoken_qa(row['audio_path'], row['language'], row['question'],  'en'))
    df["hypothesis"] = [res['content'] for res in results]
    for lang, group in df.groupby("language"):
        group.to_csv(f"results/spoken_qa/gemma4_{lang}.csv", index=False)
    
        

def main():
    # parse command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()
    print(args.model)
    df = pd.read_csv("data/Spoken QA/meta_data.csv")
   
    # check if files exist
    #get current working directory and prepend to audio path
    cwd = os.getcwd()
    
    
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(cwd,'data', x))
    
    df["file_exists"] = df["audio_path"].apply(lambda x: os.path.exists(x))

    print(df["file_exists"].value_counts())
    # filter out files that do not exist
    df = df[df["file_exists"]]

    # run inference

    if args.model == "gemma4":
        infer_gemma4(df)
    else:
        raise ValueError("Model not supported")






if __name__ == "__main__":
    main()