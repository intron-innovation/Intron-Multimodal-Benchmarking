import time
from xmlrpc import client

from google import genai
from google.genai import types
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
