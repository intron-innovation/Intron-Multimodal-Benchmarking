import re

import numpy as np
import jiwer
import string
import os
from openai import OpenAI
from whisper_normalizer.english import EnglishTextNormalizer, EnglishNumberNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from sacrebleu.metrics import BLEU, CHRF
import unicodedata
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from comet import download_model, load_from_checkpoint
    
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

bleu_single_sentence = BLEU(effective_order=True)
bleu = BLEU()
chrf = CHRF()
model_path = download_model("masakhane/africomet-mtl")
commet_model = load_from_checkpoint(model_path)

clinical = ["General-Clinical", "Clinical-Surgery", "Talk-Very-Fast-Clinical", 
            "Pre-Clinical", "Clinical-Medicine", "Pre-Clinical-INT"]
general = ["40yrs-old-and-above", "Talk-Very-Fast-Anyone", 'Transcribe-Inference',
           "Naija-News-Non-Clinical", "News-Anyone-INT", 'Transcribe-Conversation']
legal = ['Legally-Speaking', 'Transcribe-NASS', 'Transcribe-Kenya', 
         'Transcribe-South-Africa', 'Transcribe-Ghana']

inaudible_tags = ['[music] [inaudible]', '(inaudible) ', '[inaudible)', '(inaudible]',
                  '[Inaudible].', '[music]','[INAUDIBLE]',' [Inaudible]', '(Inaudible).',
                  '[Inaudible] ', '[silence]','[Silence]', '[inaudible] ', 'in aduible',
                  '(inaudible)','(Inaudible)','[Inaudible]', 'Inaudible','[inaudible]',
                  '[inaudable]','[Inaudible]','Inaudable ','Blank ', 'inaudible', 'Inaudible ', 
                  '(audio is empty)', 'noise', '(noise)', '[noise]', 'Blank'
                 ]
inaudible_tags_regex = [x.replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)') for x in inaudible_tags]
inaudible_tags_joined = "|".join(inaudible_tags_regex)
rx = re.compile(inaudible_tags_joined, re.I)
translator = str.maketrans('', '', string.punctuation)

general_filler_words = ["ah", "blah", "eh", "hmm", "huh", "hum", "mmhmm", "mm", "oh", "ohh", "uh", "uhhuh", "umhum", "uhhum", "um"]

api_key = os.getenv('OPENAI_API_KEY', "NOT_SET")

def clean_text(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    if type(text) != str:
        print(text)
        return " "

    # remove multiple spaces
    text = clean_filler_words(text)
    text = re.sub(r"\s\s+", " ", text)
    # strip trailing spaces
    text = text.strip()
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace(" comma,", ",") \
        .replace(" koma,", " ") \
        .replace(" coma,", ",") \
        .replace(" comma", " ") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",") \
        .strip()
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text

number_normalizer = EnglishNumberNormalizer()

def clean_multilingual_text(text, remove_diacritics=False, normalize_numbers=True):
    text = number_normalizer(text)
    normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
    text = text.replace(" comma,", ",") \
        .replace(" koma,", " ") \
        .replace(" coma,", ",") \
        .replace(" comma", " ") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",") \
        .replace(":", " ")\
        .replace(";", " ") \
        .replace("?", " ") \
        .replace("!", " ") \
        .replace("(", " ") \
        .replace(")", " ") \
        .replace("[", " ") \
        .replace("]", " ") \
        .replace(',', ' ') \
        .replace('.', ' ') \
        .strip()
    
    text = " ".join(text.split())
    return normalizer(text)

def clean_text_MT(
    text: str,
    lang: str = 'en',
    lowercase: bool = False,
    sacre_tokenizer: str = '13a'
) -> str:
    """
    Preprocess a list of strings for MT evaluation:
    - Unicode NFC normalization
    - Remove control/non-printing characters
    - Normalize punctuation (Moses)
    - Tokenize (Moses)
    - Optional lowercasing
    - Detokenize via SacreBLEU pipeline to get consistent spacing

    Args:
        texts: list of raw sentences (refs or preds)
        lang: language code for Moses normalizer/tokenizer
        lowercase: whether to lowercase after tokenization
        sacre_tokenizer: SacreBLEU tokenizer name ('13a', 'intl', etc.)

    Returns:
        preprocesssed list of sentences as strings
    """
    # Initialize Moses normalizer & tokenizer
    normalizer = MosesPunctNormalizer(lang=lang)
    tokenizer = MosesTokenizer(lang=lang)

    # 1) Unicode normalization (NFC)
    txt = unicodedata.normalize('NFC', text)

    # 2) Remove non-printing/control characters
    txt = re.sub(r'[\r\n\t]', ' ', txt)

    # 3) Punctuation normalization (quotes, dashes, etc.)
    txt = normalizer.normalize(txt).lower()

    return txt


def clean_text_ner(text):
    text = clean_text(text)
    text = text.translate(translator)
    return text

def clean_filler_words(text):
    text = text.replace("inaudible. ", "").replace("inaudible", "")\
        .replace(" ehm, ", " ").replace(" uh, "," ").replace(" er, "," ").replace("...", " ")
    
    tokens = re.findall(r'\b\w+\b', text)
    cleaned_tokens = [token for token in tokens if token not in general_filler_words]
    return ' '.join(cleaned_tokens)


def detect_inaudible(text):
    if (text in inaudible_tags) or (text.strip().lower() in ['inaudible', '[inaudible]', '(inaudible)']):
        return 1
    elif rx.search(text):
        return 2
    return 0


def replace_inaudible(text, pad_token=''):
    if (text in inaudible_tags) or (text.strip().lower() in ['inaudible', '[inaudible]', '(inaudible)']):
        text = pad_token
    else:
        text = re.sub(inaudible_tags_joined, pad_token, text)

    text = text.replace('[', '').replace(']', '')
    return text


text_to_digit = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": '00',
    "thousand": '000',
    "million": '000000',
    "billion": '000000000',
    "first": '1st',
    "second": '2nd',
    "third": '3rd',
    "fourth": '4th',
    "fifth": '5th',
    "sixth": '6th',
    "seventh": '7th',
    "eighth": '8th',
    "nineth": '9th',
    "tenth": '10th',
    "eleventh": '11th',
    "twelveth": '12th',
    "thirteenth": '13th',
    "fourteenth": '14th',
    "fifteenth": '15th',
    "sixteenth": '16th',
    "seventeenth": '17th',
    "eighteenth": '18th',
    "nineteenth": '19th',
    "twentieth": '20th',
    "thirtieth": '30th',
    "fortieth": '40th',
    "fiftieth": '50th',
    "sixtieth": '60th',
    "seventieth": '70th',
    "eightieth": '80th',
    "ninetieth": '90th',
    "hundredth": '00th',
    "thousandth": '000th',
    "millionth": '000000th',
    "billionth": '000000000th',
}


def text_to_numbers(text):
    text = text.split()
    return " ".join([str(text_to_digit[digit.lower()]) if digit.lower() in text_to_digit else digit for digit in text])


def strip_task_tags(s):
    if s.endswith('>'):
        return s[:s.find('<')]
    elif s.startswith('<'):
        return s[s.rfind('>')+1:]
    return s


def get_task_tags(s):
    if s.endswith('>'):
        return s[s.find('<'):]
    elif s.startswith('<'):
        return s[:s.rfind('>')+1]
    return s


def assign_domain(project_name):
    if project_name in clinical:
        return "clinical"
    elif project_name in general:
        return "general"
    elif project_name in legal:
        return "legal"
    else:
        return "general"
    
def is_accent_multiple(s):
    #print(f"accent:--{s}--")
    if len(s.split('_')) > 2:
        return 1
    elif 'pair_' in s:
        return 1
    else:
        return 0
    

def get_minority_accents(data, majority_count=5000):
    accent_counts = data.accent.value_counts().to_dict()
    print(accent_counts)
    return [accent for accent, count in accent_counts.items() if count < majority_count]

def gpt4_correcter(text):
    client = OpenAI(api_key=api_key)
    prompt = f'''You are a helpful African speech-to-text transcription assistant. Your task is to review and correct ASR transcription errors maintaining the wording of the original transcript. Consider diverse speaker accents. 
                Ensure the enhanced text mirrors the original spoken content without adding new material. DO NOT REPLACE OR ADD ANY OTHER WORDS, but fix punctuation, capitalisation and spellings. 
                The transcript is a medical conversation, therefore, also correct misspellings of medical terminologies. Your goal is to create a transcript that is accurate to the initial transcript. 
                Ignore the [blankaudio] turns. Only generate the enhanced transcript.

                Transcript:
                {text}

                Enhanced transcript:
                        '''
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
    return response.choices[0].message.content.strip()

def post_process_preds(data, correct=False, english=False, remove_diacritics=False, task="transcribe"):
    assert "hypothesis" in data.columns
    assert "reference" in data.columns

    # Standardize reference input
    refs = data["reference"].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    if task == "transcribe":
        # --- 1. UNNORMALIZED / BASIC CLEANUP METRICS ---
        if english:
            pred_basic = [clean_text(str(text)).strip() for text in data["hypothesis"]]
            ref_basic = [clean_text(str(text)).strip() for text in refs]
            # Remove commas as per your original logic
            pred_basic = [text.replace(",", "") for text in pred_basic]
            ref_basic = [text.replace(",", "") for text in ref_basic]
        else:
            pred_basic = [clean_multilingual_text(str(text), remove_diacritics).strip() for text in data["hypothesis"]]
            ref_basic = [clean_multilingual_text(str(text), remove_diacritics).strip() for text in refs]

        # Handle empty strings to avoid jiwer errors and remove periods
        pred_basic = [text.replace(".", "") if text.strip() != "" else "abcxyz" for text in pred_basic]
        ref_basic = [text.replace(".", "") if text.strip() != "" else "abcxyz" for text in ref_basic]
        
        data["pred_basic"] = pred_basic
        data["ref_basic"] = ref_basic

        # Calculate Basic Metrics
        unnorm_wer = jiwer.wer(list(data["ref_basic"]), list(data["pred_basic"]))
        unnorm_cer = jiwer.cer(list(data["ref_basic"]), list(data["pred_basic"]))
        
        print(f"Unnormalized (Cleanup) WER: {unnorm_wer * 100:.2f}%")

        # --- 2. NORMALIZED METRICS (Whisper Style) ---
        normalizer = EnglishTextNormalizer() if english else BasicTextNormalizer(remove_diacritics=remove_diacritics)
        
        if english:
            pred_norm = [normalizer(str(text)) for text in data["hypothesis"]]
            ref_norm = [normalizer(str(text)) for text in refs]
        else:
            pred_norm = [clean_multilingual_text(str(text), remove_diacritics).strip() for text in data["hypothesis"]]
            ref_norm = [clean_multilingual_text(str(text), remove_diacritics).strip() for text in refs]
        
        # Guard against empty strings after normalization
        pred_norm = [text if text.strip() != "" else "abcxyz" for text in pred_norm]
        ref_norm = [text if text.strip() != "" else "abcxyz" for text in ref_norm]
        
        data["pred_normalized"] = pred_norm
        data["ref_normalized"] = ref_norm

        norm_wer = jiwer.wer(ref_norm, pred_norm)
        norm_cer = jiwer.cer(ref_norm, pred_norm)
        
        print(f"Normalized (Whisper) WER: {norm_wer * 100:.2f}%")

        # --- 3. OPTIONAL LLM CORRECTION ---
        llm_wer = None
        if correct:
            pred_llm = [gpt4_correcter(text) for text in data["hypothesis"]]
            pred_llm = [text if text.strip() != "" else "abcxyz" for text in pred_llm]
            data["pred_llm"] = pred_llm
            llm_wer = jiwer.wer(list(refs), list(data["pred_llm"]))
            print(f"LLM Corrected WER: {llm_wer * 100:.2f}%")

        # Return all calculated metrics
        return {
            "unnormalized_wer": unnorm_wer,
            "unnormalized_cer": unnorm_cer,
            "normalized_wer": norm_wer,
            "normalized_cer": norm_cer,
            "llm_wer": llm_wer
        }
    
    return {}

def transcription_evals():
    bench_results = os.listdir("results/transcription")
    
    # 1. Map files to a structured dictionary: {language: {model: filename}}
    lang_model_map = {}
    all_models = set()
    
    for file in bench_results:
        if not file.endswith(".csv"): continue
        # Expecting format: model_language.csv
        parts = file.replace(".csv", "").split("_")
        model = parts[0]
        language = parts[1]
        
        all_models.add(model)
        if language not in lang_model_map:
            lang_model_map[language] = {}
        lang_model_map[language][model] = file

    # 2. Prepare lists to hold rows for our final DataFrames
    rows_wer = []
    rows_cer = []
    rows_unnorm_wer = []
    rows_unnorm_cer = []

    # 3. Iterate through each language (this becomes our row)
    for language, models_available in lang_model_map.items():
        # Initialize the row with the language name
        temp_wer = {"Language": language}
        temp_cer = {"Language": language}
        temp_unnorm_wer = {"Language": language}
        temp_unnorm_cer = {"Language": language}
        
        # Fill in metrics for each model (these become our columns)
        for model in all_models:
            if model in models_available:
                filename = models_available[model]
                df = pd.read_csv(os.path.join("results/transcription", filename))
                
                print(f"Evaluating {model} on {language}...")
                
                # Use your existing post_process_preds logic
                is_english = (language.lower() == "english")
                metrics = post_process_preds(
                    df, 
                    correct=False, 
                    english=is_english, 
                    remove_diacritics=(not is_english), 
                    task="transcribe"
                )
                
                # Assign to the specific model column
                temp_wer[model] = metrics["normalized_wer"]
                temp_cer[model] = metrics["normalized_cer"]
                temp_unnorm_wer[model] = metrics["unnormalized_wer"]
                temp_unnorm_cer[model] = metrics["unnormalized_cer"]
            else:
                # If a specific model hasn't run for this language, mark as NaN
                temp_wer[model] = np.nan
                temp_cer[model] = np.nan
                temp_unnorm_wer[model] = np.nan
                temp_unnorm_cer[model] = np.nan
        
        rows_wer.append(temp_wer)
        rows_cer.append(temp_cer)
        rows_unnorm_wer.append(temp_unnorm_wer)
        rows_unnorm_cer.append(temp_unnorm_cer)

    # 4. Create and Save DataFrames
    os.makedirs("evaluations/transcriptions", exist_ok=True)
    
    df_wer = pd.DataFrame(rows_wer)
    df_cer = pd.DataFrame(rows_cer)
    df_unnorm_wer = pd.DataFrame(rows_unnorm_wer)
    df_unnorm_cer = pd.DataFrame(rows_unnorm_cer)

    # Sort each by Language alphabetically
    df_wer = df_wer.sort_values(by="Language").reset_index(drop=True)
    df_cer = df_cer.sort_values(by="Language").reset_index(drop=True)
    df_unnorm_wer = df_unnorm_wer.sort_values(by="Language").reset_index(drop=True)
    df_unnorm_cer = df_unnorm_cer.sort_values(by="Language").reset_index(drop=True)

    # Save to CSV
    df_wer.to_csv("evaluations/transcriptions/transcription_wer.csv", index=False)
    df_cer.to_csv("evaluations/transcriptions/transcription_cer.csv", index=False)
    df_unnorm_wer.to_csv("evaluations/transcriptions/transcription_unnormalized_wer.csv", index=False)
    df_unnorm_cer.to_csv("evaluations/transcriptions/transcription_unnormalized_cer.csv", index=False)
    
    print("Evaluation complete. Results sorted by language and saved.")
        
def commet_score(df,src_col, ref_col, mt_col):
    data = [{
        "src": row[src_col],
        "ref": row[ref_col],
        "mt": row[mt_col]    } for index, row in df.iterrows()]
    
    
    model_output = commet_model.predict(data, batch_size=4, gpus=1)
    return model_output['system_score']

def translation_evals():
    bench_results = os.listdir("results/translation")
    
    # 1. Map files to a structured dictionary: {language: {model: filename}}
    lang_model_map = {}
    all_models = set()
    
    for file in bench_results:
        if not file.endswith(".csv"): continue
        # Expecting format: model_language.csv
        parts = file.replace(".csv", "").split("_")
        model = parts[0]
        language = parts[1]
        
        all_models.add(model)
        if language not in lang_model_map:
            lang_model_map[language] = {}
        lang_model_map[language][model] = file

    # 2. Prepare lists to hold rows for our final DataFrames
    rows_bleu = []
    rows_chrf = []
    rows_comet = []

    # 3. Iterate through each language (this becomes our row)
    for language, models_available in lang_model_map.items():
        # Initialize the row with the language name
        temp_bleu = {"Language": language}
        temp_chrf = {"Language": language}
        temp_comet = {"Language": language}
        
        # Fill in metrics for each model (these become our columns)
        for model in all_models:
            if model in models_available:
                filename = models_available[model]
                df = pd.read_csv(os.path.join("results/translation", filename))
                
                print(f"Evaluating {model} on {language}...")
                
                refs = df["reference"].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x).tolist()
                preds = df["hypothesis"].tolist()

                bleu_score = bleu.corpus_score(preds, [refs]).score
                chrf_score = chrf.corpus_score(preds, [refs]).score
                
                temp_bleu[model] = bleu_score
                temp_chrf[model] = chrf_score
                temp_comet[model] = commet_score(df, src_col='reference', ref_col='reference', mt_col='hypothesis')
            else:
                temp_bleu[model] = np.nan
                temp_chrf[model] = np.nan
                temp_comet[model] = np.nan

        
        rows_bleu.append(temp_bleu)
        rows_chrf.append(temp_chrf)
        rows_comet.append(temp_comet)

    # 4. Create and Save DataFrames
    os.makedirs("evaluations/translations", exist_ok=True)
    
    df_bleu = pd.DataFrame(rows_bleu)
    df_chrf = pd.DataFrame(rows_chrf)
    df_comet = pd.DataFrame(rows_comet)
    # Sort each by Language alphabetically
    df_bleu = df_bleu.sort_values(by="Language").reset_index(drop=True)
    df_chrf = df_chrf.sort_values(by="Language").reset_index(drop=True)
    df_comet = df_comet.sort_values(by="Language").reset_index(drop=True)
    # Save to CSV
    df_bleu.to_csv("evaluations/translations/translation_bleu.csv", index=False)
    df_chrf.to_csv("evaluations/translations/translation_chrf.csv", index=False)
    df_comet.to_csv("evaluations/translations/translation_comet.csv", index=False)

def spoken_qa_evals():
    bench_results = os.listdir("results/spoken_qa")
    
    # 1. Map files to a structured dictionary: {language: {model: filename}}
    lang_model_map = {}
    all_models = set()
    
    for file in bench_results:
        if not file.endswith(".csv"): continue
        # Expecting format: model_language.csv
        parts = file.replace(".csv", "").split("_")
        model = parts[0]
        language = parts[1]
        
        all_models.add(model)
        if language not in lang_model_map:
            lang_model_map[language] = {}
        lang_model_map[language][model] = file

    # 2. Prepare lists to hold rows for our final DataFrames
    rows_comet = []

    # 3. Iterate through each language (this becomes our row)
    for language, models_available in lang_model_map.items():
        # Initialize the row with the language name
    
        temp_comet = {"Language": language}
        
        # Fill in metrics for each model (these become our columns)
        for model in all_models:
            if model in models_available:
                filename = models_available[model]
                df = pd.read_csv(os.path.join("results/spoken_qa", filename))
                
                print(f"Evaluating {model} on {language}...")
                
                temp_comet[model] = commet_score(df, src_col = 'answer', ref_col = 'answer', mt_col = 'hypothesis')
            else:
               
                temp_comet[model] = np.nan

        
        rows_comet.append(temp_comet)

    # 4. Create and Save DataFrames
    os.makedirs("evaluations/spoken_qa", exist_ok=True)

    df_comet = pd.DataFrame(rows_comet)
    # Sort each by Language alphabetically
    df_comet = df_comet.sort_values(by="Language").reset_index(drop=True)
    # Save to CSV
    df_comet.to_csv("evaluations/spoken_qa/spoken_qa_comet.csv", index=False)
if __name__ == "__main__":
    spoken_qa_evals()
    translation_evals()
    transcription_evals()
