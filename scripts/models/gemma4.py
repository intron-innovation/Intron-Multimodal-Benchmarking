from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)


# Parse output

def transcribe_gemma(audio_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    return processor.parse_response(response)

def translate_gemma(input_audio, input_language, output_language):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": input_audio},
                {"type": "text", "text": f"Translate the following speech segment from {input_language} to {output_language}."},
                {"type": "text", "text": "Follow these specific instructions for formatting the answer:\n* Only output the translation, with no newlines."},]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    return processor.parse_response(response)


def spoken_qa(input_audio, input_language, question, output_language):

    # "Prompt is answering a question given the audio and the question. The answer should be in the output language. The question is in the input language. The audio is in the input language."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": input_audio},
                {"type": "text", "text": f"Answer the following question given the audio and the question. The answer should be in {output_language}. The question is in {input_language}. The audio is in {input_language}."},
                {"type": "text", "text": f"Question: {question}"},
                {"type": "text", "text": "Follow these specific instructions for formatting the answer:\n* Only output the translation, with no newlines."},]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    return processor.parse_response(response)