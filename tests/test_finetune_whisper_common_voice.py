"""
Main Reference: https://huggingface.co/blog/fine-tune-whisper
"""

import argparse


def main(args):
    # Set debug mode from command-line argument.
    DEBUG_MODE = args.debug

    # Map two-letter language codes to full names if necessary.
    language_mapping = {"hi": "Hindi", "ko": "Korean"}
    language_full = language_mapping.get(args.language, args.language)

    # Build dataset identifier using the provided version.
    dataset_version = args.dataset_version  # e.g., "12_0"
    dataset_id = f"mozilla-foundation/common_voice_{dataset_version}"

    """
    Load Dataset
    """
    from datasets import load_dataset, DatasetDict, Audio

    # Initialize dataset dictionary.
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        dataset_id,
        args.language,
        split="train+validation",
    )

    # Load test split for the specified language.
    common_voice["test"] = load_dataset(
        dataset_id,
        args.language,
        split="test",
    )

    print(common_voice)

    # Remove unnecessary columns.
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    print(common_voice)

    # If debug mode is enabled, reduce the dataset size for a quicker test.
    if DEBUG_MODE:
        for split in common_voice.keys():
            common_voice[split] = common_voice[split].select(range(min(len(common_voice[split]), 100)))
        print("Debug mode: Reduced dataset size to 100 samples per split.")
        # Since the evaluation set is small, it will be reduced to its original size.
        print("Evaluation set is small, it will be reduced to its original size.")
        print(common_voice)

    """
    Prepare Feature Extractor, Tokenizer and Data
    """
    # Use the model name provided as an argument.
    model_name = args.model_name

    # Load the feature extractor from the pretrained model.
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # Load the tokenizer, setting the language and task.
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language_full, task="transcribe")

    # Load the processor which combines feature extractor and tokenizer functionalities.
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name, language=language_full, task="transcribe")

    # Display the first sample from the training set.
    print(common_voice["train"][0])

    # Downsample audio from 48kHz to 16kHz.
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # Re-load and display the first audio sample to verify resampling.
    print(common_voice["train"][0])

    # Function to prepare the dataset for the model.
    def prepare_dataset(batch):
        # Load and resample audio data.
        audio = batch["audio"]

        # Compute log-Mel input features.
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # Encode the transcription to label ids.
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    # Apply the preparation function to the dataset.
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=16,
    )

    """
    Training and Evaluation
    """
    from transformers import WhisperForConditionalGeneration

    # Load the pretrained Whisper model.
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Force the model to use the desired language during generation.
    model.generation_config.language = language_full.lower()  # e.g., "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Define a custom data collator for speech sequence-to-sequence tasks.
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Process input features.
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Process labels.
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding tokens with -100.
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Remove the beginning-of-sequence token if present.
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load the Word Error Rate (WER) metric.
    import evaluate

    metric = evaluate.load("wer")

    # Function to compute evaluation metrics.
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad token id.
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels.
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Set up training arguments for 1 epoch training.
    from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{args.model_name.replace('/', '-')}-{args.language}",  # Dynamically set output directory.
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=1,  # Train for one epoch to avoid overfitting.
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8 if not DEBUG_MODE else 2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000 if not DEBUG_MODE else 10,
        eval_steps=1000 if not DEBUG_MODE else 10,
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
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Save the processor (not trainable).
    processor.save_pretrained(training_args.output_dir)

    # Start training.
    trainer.train()

    # Push the model to the Hugging Face Hub.
    kwargs = {
        "dataset_tags": dataset_id,
        "dataset": f"Common Voice {dataset_version}",  # A 'pretty' name for the dataset.
        "dataset_args": f"config: {args.language}, split: test",
        "language": args.language,
        "model_name": f"{args.model_name} {language_full} - Sanchit Gandhi",
        "finetuned_from": model_name,
        "tasks": "automatic-speech-recognition",
    }
    if args.push_to_hub:
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Common Voice")
    parser.add_argument("--dataset_version", type=str, default="12_0", help="Common Voice dataset version (e.g., 12_0)")
    parser.add_argument("--language", type=str, default="ko", help="Language code for dataset and model (e.g., 'ko' for Korean)")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3-turbo", help="Pretrained model identifier to use for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced dataset size and fewer training steps")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to the Hugging Face Hub")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size per device for training")
    args = parser.parse_args()

    main(args)
