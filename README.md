# LLM Fine-Tuning Lab

Fine-tune TinyLlama-1.1B on instruction data using LoRA — on a MacBook, no GPU required.

---

## Why This Project

Every other project in this portfolio uses an LLM as a black box: send a prompt, get a response.

This project goes **inside** the model and changes how it behaves.

---

## Upgrade: v1 → v2

| | v1 | v2 |
|---|---|---|
| Model | distilgpt2 (82M) | TinyLlama-1.1B |
| Method | Full fine-tuning | LoRA (trains 0.38%) |
| Dataset | Custom texts | Stanford Alpaca (52k) |
| Trainable params | 82M | ~4M |
| Adapter size | ~300MB | ~50MB |

---

## What LoRA Does

```
Without LoRA:
  Update all 1,100,000,000 parameters → needs A100 GPU

With LoRA (r=16):
  Freeze base model
  Add small adapter matrices to attention layers
  Train only ~4,000,000 parameters (0.38%)
  Runs on a MacBook
```

---

## Project Structure

```
llm-fine-tuning/
├── src/
│   ├── trainer.py     — LLMTrainer: TinyLlama + LoRA fine-tuning
│   └── evaluator.py   — ModelEvaluator: perplexity comparison
├── data/
│   ├── prepare.py     — download Alpaca, convert to ChatML format
│   └── sample.json    — 200-example sample for quick testing
├── notebooks/         — walkthrough notebooks
├── models/            — saved LoRA adapters (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
pip install -r requirements.txt
```

### Step 1 — Prepare dataset

```bash
python3 data/prepare.py --sample 500
```

### Step 2 — Fine-tune

```python
import json
from src.trainer import LLMTrainer

with open("data/sample.json") as f:
    data = json.load(f)

trainer = LLMTrainer(lora_r=16, lora_alpha=32)
losses = trainer.train(
    train_data=data[:180],
    eval_data=data[180:],
    output_dir="models/lora_adapter",
    epochs=3,
)
```

### Step 3 — Generate

```python
response = trainer.generate(
    "<|system|>\nYou are a helpful assistant.</s>\n"
    "<|user|>\nExplain what machine learning is.</s>\n"
    "<|assistant|>\n"
)
print(response)
```

### Step 4 — Evaluate perplexity

```python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator(trainer.tokenizer, trainer.model.device)
# compare base vs fine-tuned perplexity
```

---

## LoRA Configuration

| Parameter | Value | Meaning |
|---|---|---|
| `r` (rank) | 16 | Size of adapter matrices |
| `lora_alpha` | 32 | Scaling factor |
| `target_modules` | q_proj, v_proj | Attention layers to adapt |
| `lora_dropout` | 0.05 | Regularization |
| Trainable params | ~4M | 0.38% of total |

---

## Stack

Python · PyTorch · HuggingFace Transformers · PEFT · Datasets · pytest

---

## What I Learned

LoRA is an approximation, not a shortcut.
Instead of learning ΔW (d×d), learn A (d×r) and B (r×d) where r << d.
Their product approximates ΔW with far fewer parameters.

Adapter weights are tiny.
Full TinyLlama is ~2.2GB. The LoRA adapter is ~50MB.
You ship the adapter, not the model.

Data format matters more than training.
TinyLlama uses ChatML tokens. Wrong format = garbage output regardless of training quality.

---

## Related Projects

- [llm-evaluation-playground](https://github.com/Honaxen/llm-evaluation-playground) — evaluating LLM outputs
- [multi-tool-agent](https://github.com/Honaxen/multi-tool-agent) — using LLMs as black boxes

---

## Author

[Honaxen](https://github.com/Honaxen)