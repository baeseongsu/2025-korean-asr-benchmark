# !pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
# !pip install qwen-omni-utils
# !pip install openai
# !pip install flash-attn --no-build-isolation

from qwen_omni_utils import process_mm_info

import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

model_path = "Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


# @title inference function
def inference(audio_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": audio_path},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


import librosa

from io import BytesIO
from urllib.request import urlopen

from IPython.display import Audio

from IPython.display import display

audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
prompt = "Transcribe the English audio into text without any punctuation marks."

audio = librosa.load(BytesIO(urlopen(audio_path).read()), sr=16000)[0]
display(Audio(audio, rate=16000))

## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])


audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/BAC009S0764W0121.wav"
prompt = "请将这段中文语音转换为纯文本，去掉标点符号。"

audio = librosa.load(BytesIO(urlopen(audio_path).read()), sr=16000)[0]
display(Audio(audio, rate=16000))

## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])


audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/10000611681338527501.wav"
prompt = "Transcribe the Russian audio into text without including any punctuation marks."

audio = librosa.load(BytesIO(urlopen(audio_path).read()), sr=16000)[0]
display(Audio(audio, rate=16000))

## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/7105431834829365765.wav"
prompt = "Transcribe the French audio into text without including any punctuation marks."

audio = librosa.load(BytesIO(urlopen(audio_path).read()), sr=16000)[0]
display(Audio(audio, rate=16000))

## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])
