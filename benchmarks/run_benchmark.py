#!/usr/bin/env python
"""
ASR Benchmarking Script with Qwen2.5-Omni Integration and Selective Caching
--------------------------------------------------------------------------
This script benchmarks ASR models from Hugging Face using pipeline().
It supports:
  - Whisper models (e.g., openai/whisper-large-v3)
  - The Qwen2.5-Omni model (using a custom inference call)
  - Korean, English, Chinese, Russian, and French languages
  - Evaluation metrics: WER and CER

The script implements selective inference via caching. For each (model, dataset, language)
combination paired with evaluation settings, it computes a unique hash. If results for that
hash already exist, it skips re-evaluation and loads the cached result; otherwise, it runs inference and updates the cache.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import time
import random
import tempfile
import json
import hashlib
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
from datasets import load_dataset, Dataset
import evaluate
from transformers import pipeline

# Standard audio processing import for writing temporary WAV files.
import soundfile as sf

# Normalizer import.
from normalizer import EnglishTextNormalizer

# Colorized logging setup.
from colorama import Fore, Style, init as colorama_init

# -------------------------------------------------
# Qwen ASR Imports
# -------------------------------------------------
from qwen_omni_utils import process_mm_info

# NOTE: The codes of Qwen2.5-Omni on Hugging Face Transformers are in pull request stage and not merged into the main branch yet. Therefore, you may need to build from source to use it with command:
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor


# -------------------------------------------------
# Colorized Logging
# -------------------------------------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        msg = super().format(record)
        return f"{color}{msg}{reset}"


colorama_init(autoreset=True)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s: %(message)s"))
root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
if len(root_logger.handlers) > 1:
    del root_logger.handlers[0]


# -------------------------------------------------
# Argument Parsing and Seed Setting
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    default_results_path = os.path.join(script_dir, "eval_results.json")  # Absolute path to results file

    parser = argparse.ArgumentParser(description="ASR Benchmarking Script")
    parser.add_argument("--split-percent", type=str, default="5%", help="Dataset test split percentage to use (e.g., 5%%)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda' or 'cpu'). If not set, auto-detects CUDA.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing audio samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--normalize", action="store_true", help="Apply text normalization before computing metrics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode to show prediction differences")
    parser.add_argument("--results-file", type=str, default=default_results_path, help="File to save or load evaluation results")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# -------------------------------------------------
# Generate a Unique Evaluation Hash
# -------------------------------------------------
def generate_evaluation_hash(model_name: str, dataset_name: str, language: str, evaluation_settings: Dict[str, Any]) -> str:
    # Sort the evaluation settings keys to ensure consistent ordering.
    settings_str = ",".join(f"{k}={evaluation_settings[k]}" for k in sorted(evaluation_settings))
    config_str = f"{model_name}|{dataset_name}|{language}|{settings_str}"
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


# -------------------------------------------------
# Qwen Inference Function (as provided)
# -------------------------------------------------
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
    # Use the Qwen processor to apply a chat template and generate inputs.
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.info(f"Inference chat template: {text}")
    # Process multimedia information (audio, image, video) for the input.
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = qwen_processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(qwen_model.device).to(qwen_model.dtype)
    # Generate output and trim the input portion from each sequence.
    generated_ids = qwen_model.generate(**inputs, use_audio_in_video=True, return_audio=False)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# -------------------------------------------------
# Custom Pipeline for Qwen Model
# -------------------------------------------------
class QwenASRPipeline:
    def __init__(self, model, processor, prompt, sys_prompt):
        """
        Initializes the custom Qwen pipeline.
        :param model: Qwen2.5-Omni model
        :param processor: Qwen2.5-Omni processor
        :param prompt: Language-specific transcription prompt.
        :param sys_prompt: System prompt for ASR inference.
        """
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.sys_prompt = sys_prompt

    def __call__(self, audio_arrays: List[np.ndarray]):
        results = []
        for audio_array in audio_arrays:
            # Write the NumPy audio array to a temporary WAV file.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio_array, 16000)  # assumes 16 kHz sample rate
                tmp_path = tmpfile.name
            # Call the Qwen inference function with the temporary file.
            transcription = inference(tmp_path, self.prompt, self.sys_prompt)
            results.append({"text": transcription})
            os.remove(tmp_path)
        return results


# -------------------------------------------------
# Metrics Computation
# -------------------------------------------------
def compute_cer(prediction: str, reference: str) -> float:
    n, m = len(reference), len(prediction)
    d = np.zeros((n + 1, m + 1), dtype=np.uint32)
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / n if n > 0 else 0


def transcribe_batch_with_pipeline(batch: Dict[str, Any], asr_pipeline) -> Dict[str, Any]:
    audio_arrays = [sample["array"] for sample in batch["audio"]]
    results = asr_pipeline(audio_arrays)  # Process the batch in one call.
    transcriptions = [res["text"] for res in results]
    batch["pred_str"] = transcriptions
    return batch


def get_datasets_for_lang(datasets: Dict[str, Dataset], lang: str) -> List[Tuple[str, Dataset]]:
    matching = []
    for key, dataset in datasets.items():
        if key.endswith(f"/{lang}"):
            matching.append((key, dataset))
    if not matching:
        raise ValueError(f"No dataset found for language '{lang}'.")
    return matching


# -------------------------------------------------
# Evaluation Function (with Selective Inference)
# -------------------------------------------------
def evaluate_model(
    model_name: str,
    config: Dict[str, Any],
    device: torch.device,
    datasets: Dict[str, Dataset],
    wer_metric,
    cer_metric,
    use_custom_cer: bool,
    batch_size: int,
    verbose: bool,
    normalize: bool,
    normalizer_instance,
    evaluation_settings: Dict[str, Any],
    cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results = []
    logging.info(f"Evaluating model: {model_name}")

    for lang in config["supported_langs"]:
        try:
            datasets_list = get_datasets_for_lang(datasets, lang)
        except ValueError as e:
            logging.error(e)
            continue

        # Set up the ASR pipeline based on model type.
        if config.get("type") == "qwen":
            global qwen_model, qwen_processor
            model_path = "Qwen/Qwen2.5-Omni-7B"
            qwen_model = Qwen2_5OmniModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            qwen_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            # Define a language-specific prompt.
            if lang == "en":
                prompt = "Transcribe the English audio into text without any punctuation marks."
            elif lang == "zh":
                prompt = "请将这段中文语音转换为纯文本，去掉标点符号。"
            elif lang == "ru":
                prompt = "Transcribe the Russian audio into text without including any punctuation marks."
            elif lang == "fr":
                prompt = "Transcribe the French audio into text without including any punctuation marks."
            elif lang == "ko":
                prompt = "Transcribe the Korean audio into text without including any punctuation marks."
            else:
                prompt = "Transcribe the audio into text."
            asr_pipeline = QwenASRPipeline(
                qwen_model,
                qwen_processor,
                prompt,
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            )
        else:
            try:
                from transformers import WhisperProcessor

                proc_whisper = WhisperProcessor.from_pretrained(model_name)
                forced_decoder_ids = proc_whisper.get_decoder_prompt_ids(language=lang)
            except Exception as e:
                logging.error(f"Failed to load WhisperProcessor for {model_name}: {e}")
                forced_decoder_ids = None

            try:
                device_index = device.index if device.type == "cuda" and device.index is not None else (0 if device.type == "cuda" else -1)
                # Set generation kwargs using forced_decoder_ids if available
                gen_kwargs = {"forced_decoder_ids": forced_decoder_ids} if forced_decoder_ids is not None else {}

                # If using ghost613 model, adopt generation_config from openai/whisper-large-v3-turbo
                if model_name == "ghost613/whisper-large-v3-turbo-korean":
                    from transformers import GenerationConfig

                    generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3-turbo")
                    gen_kwargs["generation_config"] = generation_config
                    gen_kwargs["max_length"] = 1000

                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device_index,
                    batch_size=batch_size,
                    generate_kwargs=gen_kwargs,
                )

                asr_pipeline.model.generation_config.input_ids = asr_pipeline.model.generation_config.forced_decoder_ids
                asr_pipeline.model.generation_config.forced_decoder_ids = None

            except Exception as e:
                logging.error(f"Failed to load pipeline for language {lang}: {e}")
                continue

        # Process each dataset for the current language.
        for dataset_name, dataset in datasets_list:
            # Generate a unique key for this (model, dataset, language) configuration.
            config_key = generate_evaluation_hash(model_name, dataset_name, lang, evaluation_settings)
            # If the key is in cache, skip evaluation.
            if config_key in cache:
                logging.info(f"Skipping evaluation for {model_name} {dataset_name} {lang} (cached).")
                results.append(cache[config_key])
            else:
                start_time = time.time()
                logging.info(f"Processing language: {lang} on dataset: {dataset_name}")
                transcribed_dataset = dataset.map(
                    lambda batch: transcribe_batch_with_pipeline(batch, asr_pipeline),
                    batched=True,
                    batch_size=batch_size,
                )
                references = transcribed_dataset["text"]
                predictions = transcribed_dataset["pred_str"]

                if normalize and normalizer_instance is not None:
                    references = [normalizer_instance(r) for r in references]
                    predictions = [normalizer_instance(p) for p in predictions]

                if verbose:
                    for i in range(min(5, len(references))):
                        logging.info(f"[Verbose] {lang} Example {i+1} from {dataset_name}:\n  GT:   {references[i]}\n  Pred: {predictions[i]}")

                score_wer = wer_metric.compute(predictions=predictions, references=references)
                score_cer = (
                    cer_metric.compute(predictions=predictions, references=references) if not use_custom_cer else sum(compute_cer(p, r) for p, r in zip(predictions, references)) / len(predictions)
                )

                elapsed = time.time() - start_time
                result_obj = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "language": lang,
                    "WER": score_wer,
                    "CER": score_cer,
                    "time_sec": elapsed,
                }
                results.append(result_obj)
                cache[config_key] = result_obj
    return results


# -------------------------------------------------
# Main Function with Selective Inference
# -------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.getLogger().setLevel(numeric_level)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}, GPU: {torch.cuda.get_device_name(device_index)}")

    normalizer_instance = EnglishTextNormalizer() if args.normalize else None

    # Define model configurations (here using Qwen as an example).
    model_configs = {
        # "Qwen/Qwen2.5-Omni-7B": {"supported_langs": ["en", "zh", "ru", "fr", "ko"], "type": "qwen"},
        "ghost613/whisper-large-v3-turbo-korean": {"supported_langs": ["ko"], "type": "whisper"},
        "openai/whisper-large-v3-turbo": {"supported_langs": ["ko", "en"], "type": "whisper"},
        "openai/whisper-large-v3": {"supported_langs": ["ko", "en"], "type": "whisper"},
    }

    logging.info("Loading datasets...")
    datasets = {
        "Bingsu_zeroth-korean/ko": load_dataset("Bingsu/zeroth-korean", split="test"),
        # "kresnik_zeroth_korean/ko": load_dataset("kresnik/zeroth_korean", split="test"),
        # Add more datasets as needed.
    }

    wer_metric = evaluate.load("wer")
    try:
        cer_metric = evaluate.load("cer")
        use_custom_cer = False
    except Exception:
        logging.warning("CER metric not found in `evaluate`, using fallback.")
        use_custom_cer = True

    # Load existing cached evaluation results, if any.
    if os.path.exists(args.results_file):
        logging.info(f"Loading cached evaluation results from {args.results_file}")
        with open(args.results_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Define key evaluation settings.
    evaluation_settings = {"seed": args.seed, "batch_size": args.batch_size, "normalize": args.normalize}

    all_results = []
    overall_start = time.time()
    for model_name, config in model_configs.items():
        results = evaluate_model(
            model_name,
            config,
            device,
            datasets,
            wer_metric,
            cer_metric,
            use_custom_cer,
            args.batch_size,
            args.verbose,
            args.normalize,
            normalizer_instance,
            evaluation_settings,
            cache,
        )
        all_results.extend(results)

        # Release model and clear GPU memory
        del asr_pipeline
        if "qwen_model" in globals():
            del qwen_model
        if "qwen_processor" in globals():
            del qwen_processor
        torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_start
    logging.info(f"Total evaluation time for new computations: {overall_elapsed:.2f} seconds")

    # Save the updated cache to disk.
    with open(args.results_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Print summary of results.
    logging.info("Summary of Results:")
    header = f"{'Model':50s} | {'Dataset':40s} | {'Language':10s} | {'WER (%)':10s} | {'CER (%)':10s} | {'Time (sec)':10s}"
    logging.info(header)
    logging.info("-" * len(header))
    for res in all_results:
        line = f"{res['model']:50s} | {res['dataset']:40s} | {res['language']:10s} | {res['WER']*100:10.2f}% | {res['CER']*100:10.2f}% | {res['time_sec']:10.2f}"
        logging.info(line)
    if overall_elapsed < 1:
        logging.info("Evaluation loaded entirely from cache.")
    else:
        logging.info(f"Total evaluation time for new computations: {overall_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
