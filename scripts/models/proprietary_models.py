import json
import pathlib
import time
from xmlrpc import client

from google import genai
from google.genai import types
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.transcription import TranscriptionClient
from azure.ai.transcription.models import TranscriptionContent, TranscriptionOptions
from pydub import AudioSegment
import requests
# load environment variables from .env file
from dotenv import load_dotenv
import os   
import base64
from openai import OpenAI
load_dotenv()

lang_map = {
    "afrikaans": "af",
    "akan": "ak",
    "amharic": "am",
    "arabic": "ar",
    "english": "en",
    "french": "fr",
    "hausa": "ha",
    "igbo": "ig",
    "kinyarwanda": "rw",
    "pedi": "nso",
    "sesotho": "st",
    "shona": "sn",
    "swahili": "sw",
    "tswana": "tn",
    "xhosa": "xh",
    "yoruba": "yo",
    "zulu": "zu"
}
language_locales = {
                "af": "af-ZA",
                "am": "am-ET",
                "ar": "ar-MA",
                "en": "en-NG",
                "fr": "fr-FR",
                "sw": "sw-KE",
                "zu": "zu-ZA"
            }
def ensure_mono(file_path):
    # 1. Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # 2. Check number of channels
    # 1 = Mono, 2 = Stereo
    channels = audio.channels
    print(f"Original channels: {channels}")

    if channels > 1:
        print("Converting stereo to mono...")
        # 3. Convert to mono
        mono_audio = audio.set_channels(1)
        
        # 4. Export the new file
        #write new directory if it does not exist
        new_dir = os.path.join(os.path.dirname(file_path), "mono")
        os.makedirs(new_dir, exist_ok=True)
        output_path = os.path.join(new_dir, "mono_version_" + os.path.basename(file_path))
        mono_audio.export(output_path, format="wav")
        print(f"Success! Saved to: {output_path}")
        return output_path
    else:
        print("Audio is already mono.")
        return file_path     

def transcribe_google_med_stt(audio_file_path):
    api_key = os.getenv("GOOGLE_MED_STT_API_KEY")
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"

    audio_file_path = ensure_mono(audio_file_path)
    # 2. Read and encode audio to Base64
    with open(audio_file_path, "rb") as audio_file:
        audio_content = base64.b64encode(audio_file.read()).decode("utf-8")
    payload = {
        "config": {
            "languageCode": "en-US",
            "model": "medical_conversation",
            "enableAutomaticPunctuation": True
        },
        "audio": {
            "content": audio_content
        }
    }

    response = requests.post(url, data=json.dumps(payload))

    if response.status_code == 200:
        results = response.json().get("results", [])
        for result in results:
            return result["alternatives"][0]["transcript"]
    else:
        print(f"Error {response.status_code}: {response.text}")

def gemini_translate(audio_path, source_language, target_language = "en", model_name = "gemini-3-flash-preview"):

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_2"))
    audio_file = client.files.upload(file=audio_path)
    # Generate the translation
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                audio_file,
                f"Translate this african accented audio from {source_language} to {target_language}. Provide the translation as text. output only the translation without any additional text. Do not include speaker labels or timestamps. Only provide the transcribed text. do not diarize the audio. do not include any timestamps. do not include any speaker labels, do not diarize the audio. Only provide the translated text."
            ]
        )
        translation = response.text
  
        print(translation)
        return translation
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"

    

def gpt4o_transcribe(audio_path):
    client = OpenAI()
    try:
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model="gpt-4o-transcribe",
                response_format="text"  # or "json"
            )
        return transcript
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"

def qwen3_translate(audio_path, source_language, target_language = "en"):
    client = OpenAI()
    try:
        client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore
        )
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        completion = client.chat.completions.create(
            model="qwen3-livetranslate-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/wav;base64,{audio_base64}",
                                "format": "wav",
                            },
                        }
                    ],
                }
            ],
                    stream=False,
                    extra_body={"translation_options": {"source_lang": lang_map.get(source_language, "en"), "target_lang": lang_map.get(target_language, "en")}},
                )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"
def qwen3_transcribe(audio_path, source_language):
    client = OpenAI()
    try:
        client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore
        )
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        completion = client.chat.completions.create(
            model="qwen3-asr-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/wav;base64,{audio_base64}"
                            }
                        }
                    ]
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"    
def gpt4o_audio_translate(audio_path, source_language, target_language = "en"):
    client = OpenAI()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text"],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                },
                {
                    "type": "text",
                    "text": f"Translate this african accented audio from {source_language} to {target_language}. Provide the translation as text. output only the translation without any additional text. do not include any timestamps. do not include any speaker labels, do not diarize the audio. Only provide the translated text."
                }
            ]
        }
        ],
        ) 
        translation = response.choices[0].message.content
        
        return translation
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"

def gemini_transcribe(audio_path, source_language, model_name = "gemini-3-flash-preview"):

    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    
    audio_file = client.files.upload(file=audio_path)
    # Generate the transcription
    response = client.models.generate_content(
        model=model_name,
        contents=[
            audio_file,
            f"Transcribe this african accented audio in {source_language} . Provide the transcription as text. output only the transcription without any additional text. Do not include speaker labels or timestamps. Only provide the transcribed text. do not diarize the audio. do not include any timestamps. do not include any speaker labels, do not diarize the audio. Only provide the transcribed text."
        ]
    )

    
    transcription = response.text
    return transcription
def transcribe_audio_azure(audio_path, language):
    # Replace with your own subscription key and region
    endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")
    api_key = os.environ.get("AZURE_SPEECH_KEY")

    if api_key:
        credential = AzureKeyCredential(api_key)
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
    try:
        audio_file_path = pathlib.Path(audio_path)
        client = TranscriptionClient(endpoint=endpoint, credential=credential)
        # Open and read the audio file
        with open(audio_file_path, "rb") as audio_file:
            # Create transcription options
            # create locales using the language and langauge mapping
            # Add more language mappings as needed
            first_ap = lang_map.get(language.lower(), "")
            locale = language_locales.get(first_ap.lower(), None)  # Default to English if language not found
            print(f"Using locale {locale} for language {language}")
            if not locale:
                print(f"Locale not found for language {language}. Skipping transcription.")
                return ""
            options = TranscriptionOptions(
                locales= [locale]  
            )

            request_content = TranscriptionContent(definition=options, audio=audio_file)

            result = client.transcribe(request_content)

        return result.combined_phrases[0].text
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"


def translate_audio_azure(audio_path, source_language, target_language='english'):
    # Replace with your own subscription key and region
    region = os.environ.get("AZURE_SPEECH_REGION")
    api_key = os.environ.get("AZURE_SPEECH_KEY")

    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=api_key,
        region=region
    )

    from_lang = language_locales.get(lang_map.get(source_language.lower(), "").lower(), "")
    to_lang = language_locales.get(lang_map.get(target_language.lower(), "").lower(), "")
    print(from_lang, to_lang)
    if from_lang == "" or to_lang == "":
        return "ERROR"
    translation_config.speech_recognition_language = from_lang
    
    translation_config.add_target_language(to_lang)

    # Audio input
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config,
        audio_config=audio_config
    )
    try:
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.TranslatedSpeech:
            
            return result.translations[to_lang]

        else: 
            print(f"Error processing {audio_path}: {result.reason}")
            return "ERROR"  
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return "ERROR"
