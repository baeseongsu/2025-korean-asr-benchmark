"""
Main Reference: https://huggingface.co/blog/fine-tune-whisper
This script fine-tunes the Whisper model on a custom ASR dataset.
It supports datasets with different column names by letting you specify:
  - dataset_id and an optional dataset_config,
  - the name of the column containing the audio data,
  - the name of the column containing transcription text.
"""

import argparse


def main(args):
    DEBUG_MODE = args.debug

    # Map two-letter language codes to full names if necessary.
    language_mapping = {"hi": "Hindi", "ko": "Korean"}
    language_full = language_mapping.get(args.language, args.language)

    # Use the provided dataset identifier.
    dataset_id = args.dataset_id

    """
    Load Dataset
    """
    from datasets import load_dataset, DatasetDict, Audio

    dataset = DatasetDict()

    # Load dataset splits (using config if provided).
    if args.dataset_config:
        dataset["train"] = load_dataset(dataset_id, args.dataset_config, split="train")
        dataset["test"] = load_dataset(dataset_id, args.dataset_config, split="test")
    else:
        dataset["train"] = load_dataset(dataset_id, split="train")
        dataset["test"] = load_dataset(dataset_id, split="test")

    print("Loaded dataset splits:")
    print(dataset)

    # Optional: Remove columns not needed (if desired you can add logic here).

    # If debug mode is enabled, reduce the dataset size.
    if DEBUG_MODE:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(min(len(dataset[split]), 100)))
        print("Debug mode: Reduced dataset size to 100 samples per split.")

    """
    Prepare Feature Extractor, Tokenizer and Data
    """
    model_name = args.model_name

    # Load the feature extractor.
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # Load the tokenizer with the desired language and task.
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language_full, task="transcribe")

    # Load the processor that combines the feature extractor and tokenizer.
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name, language=language_full, task="transcribe")

    print("First training sample (before resampling):")
    print(dataset["train"][0])

    # Downsample the audio column to 16kHz.
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=16000))

    print("First training sample (after resampling):")
    print(dataset["train"][0])

    # Function to prepare dataset by extracting features and encoding text.
    def prepare_dataset(batch):
        audio = batch[args.audio_column]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[args.transcription_column]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=16,
    )

    """
    Training and Evaluation
    """
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Set generation configuration.
    model.generation_config.language = language_full.lower()  # e.g., "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Define a data collator that pads inputs and labels separately.
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load evaluation metric.
    import evaluate

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Set up training arguments for one epoch.
    from transformers import Seq2SeqTrainingArguments

    OUTPUT_DIR = f"./{args.model_name.replace('/', '-')}-{args.language}-TEST"

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=5,  # One epoch to mitigate overfitting.
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8 if not DEBUG_MODE else 2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100 if not DEBUG_MODE else 10,
        eval_steps=25 if not DEBUG_MODE else 10,
        logging_steps=25 if not DEBUG_MODE else 5,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Save the processor.
    processor.save_pretrained(training_args.output_dir)

    # Begin training.
    trainer.train()

    # Push to hub if enabled.
    kwargs = {
        "dataset_tags": dataset_id,
        "dataset": dataset_id,
        "dataset_args": f"transcription column: {args.transcription_column}",
        "language": args.language,
        "model_name": f"{args.model_name} {language_full} - Fine-tuned",
        "finetuned_from": model_name,
        "tasks": "automatic-speech-recognition",
    }
    if args.push_to_hub:
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a custom ASR dataset")
    parser.add_argument("--dataset_id", type=str, default="Bingsu/zeroth-korean", help="Dataset identifier on Hugging Face (e.g., 'Bingsu_zeroth-korean/ko')")
    parser.add_argument("--dataset_config", type=str, default=None, help="Optional dataset configuration (if required)")
    parser.add_argument("--audio_column", type=str, default="audio", help="Name of the column containing the audio data")
    parser.add_argument("--transcription_column", type=str, default="text", help="Name of the column containing transcription text")
    parser.add_argument("--language", type=str, default="ko", help="Language code for model and tokenizer (e.g., 'ko' for Korean)")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3-turbo", help="Pretrained model identifier to use for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced dataset size and fewer training steps")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to the Hugging Face Hub after training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size per device for training")
    args = parser.parse_args()

    main(args)
