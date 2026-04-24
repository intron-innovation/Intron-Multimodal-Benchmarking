import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


def transcribe_qwen3omni(audio_path, model, processor):
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
        ],
    },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    input_ids = inputs["input_ids"]

    attention_mask = torch.ones_like(input_ids)
    attention_mask = torch.ones_like(input_ids)

    # If padding was applied, detect trailing padding by length
    for i in range(input_ids.size(0)):
        # find first padding index (from the right)
        seq = input_ids[i]
        non_pad = (seq != processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad) > 0:
            last_real = non_pad[-1].item()
            attention_mask[i, last_real+1:] = 0

    inputs["attention_mask"] = attention_mask
    text_ids, audio = model.generate(**inputs, 
                                    pad_token_id=processor.tokenizer.pad_token_id,
                                    thinker_return_dict_in_generate=True,
                                    use_audio_in_video=False)

    text = processor.batch_decode(
    text_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
    )
    return text[0]

def translate_qwen3omni(input_audio, input_language, output_language, processor, model):
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": input_audio},
        {"type": "text", "text": f"Translate the following speech segment from {input_language} to {output_language}."},
        {"type": "text", "text": "Follow these specific instructions for formatting the answer:\n* Only output the translation, with no newlines."}
               ],
    },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(**inputs, 
                                    speaker="Ethan", 
                                    thinker_return_dict_in_generate=True,
                                    use_audio_in_video=False)

    text = processor.batch_decode(
    text_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
    )
    return text[0]


def load_model(model_name):
   
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
    return model, processor
