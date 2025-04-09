#!/usr/bin/env python
"""
Test Script for ghost613/whisper-large-v3-turbo-korean with Bingsu Zeroth-Korean Dataset
---------------------------------------------------------------------------------------
This script loads the test split from the "Bingsu/zeroth-korean" dataset, selects one sample,
and uses the ghost613 ASR pipeline to transcribe the sample audio.
"""

from transformers import pipeline, WhisperProcessor, GenerationConfig
from datasets import load_dataset
import torch

# Set device: use GPU if available, otherwise CPU (-1)
device = 0 if torch.cuda.is_available() else -1

model_name = "ghost613/whisper-large-v3-turbo-korean"
language = "ko"  # Specify Korean

# Load the Whisper processor and compute forced_decoder_ids for the desired language
try:
    processor = WhisperProcessor.from_pretrained(model_name)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language)
except Exception as e:
    forced_decoder_ids = None

# Load the generation config (using the openai/whisper-large-v3-turbo config)
generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3-turbo")
print("Loaded generation_config:")
print(generation_config)

# Set generation arguments if forced_decoder_ids are available
gen_kwargs = {"forced_decoder_ids": forced_decoder_ids} if forced_decoder_ids is not None else {}
gen_kwargs["generation_config"] = generation_config
gen_kwargs["max_length"] = 1000

# Set torch_dtype based on device availability
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize the ASR pipeline with the ghost613 model
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    config="openai/whisper-large-v3-turbo",
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=1,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs=gen_kwargs,
)

# Load the test split of the Bingsu zeroth-korean dataset
dataset = load_dataset("Bingsu/zeroth-korean", split="test")

# Select the first sample
sample = dataset[0]
audio_array = sample["audio"]["array"]
ground_truth = sample["text"]

# Run transcription on the sample audio
result = asr_pipeline(audio_array)

print("Ground truth:", ground_truth)
print("Transcription:", result["text"])
