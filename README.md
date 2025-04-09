# 2025-korean-asr-benchmark

A personal benchmarking tool for ASR models across multilingual datasets.

---

## Features

- [x] Supports Whisper & Qwen2.5-Omni
- [x] Languages: Korean, English, Chinese, Russian, French
- [x] Computes WER & CER
- [x] Caching for faster re-runs

---

## Models

- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`
- `ghost613/whisper-large-v3-turbo-korean`
- `Qwen/Qwen2.5-Omni-7B` *(requires custom Transformers build)*

---

## Dataset

- `Bingsu/zeroth-korean`  
(Add more via `load_dataset()`)

---

## Run

```bash
python benchmarks/run_benchmark.py --normalize --verbose
```

---

Let me know if you want this in Korean too or if you’d like a badge-style header!
python benchmarks/run_benchmark_v3.py