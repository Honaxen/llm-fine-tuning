# LLM Fine-Tuning

Fine-tuning a small language model from scratch.
Understanding what happens inside the model when we adapt it to new data.

---

## Why Fine-Tuning?

Pre-trained models know a lot — but not everything.
Fine-tuning adapts a general model to a specific domain or task.

This project fine-tunes DistilGPT-2 on custom text data
and measures how the model changes before and after training.

---

## Model

**DistilGPT-2** — a distilled version of GPT-2
- runs on Apple Silicon (MPS)
- 6 transformer layers, 12 attention heads
- 768 hidden dimensions
- Same architecture as GPT-2, 40% fewer parameters

---

## Project Structure

```
llm-fine-tuning/
├── notebooks/
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── trainer.py
│   └── evaluator.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── README.md
```

---

## Stack

Python · PyTorch · HuggingFace Transformers · Apple MPS

---

## What I Learned

TBD — will be updated after all notebooks are complete.

---

## Author

[Honaxen](https://github.com/Honaxen)