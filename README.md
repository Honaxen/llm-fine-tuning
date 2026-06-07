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
│   └── 05_mlflow_tracking.ipynb
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

Python · PyTorch · HuggingFace Transformers · MLflow · Apple MPS

---

## What I Learned

Fine-tuning works — but small datasets have a cost.

ML text perplexity dropped 86% after training on 41 sentences.
The model learned ML vocabulary and sentence patterns quickly.

But general text perplexity increased 54% — catastrophic forgetting.
When you train on new data, the model partially overwrites what it knew before.

The lesson: fine-tuning is a trade-off between domain adaptation and general knowledge.
More data, lower learning rate, and mixed training reduce forgetting.

Perplexity is the right metric for evaluating language model understanding.
It measures surprise — a model that understands the domain is less surprised by domain text.

---

## Author

[Honaxen](https://github.com/Honaxen)