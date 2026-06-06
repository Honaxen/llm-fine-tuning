"""
evaluator.py
------------
Evaluation metrics for fine-tuned language models.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ModelEvaluator:
    """Evaluate language model quality using perplexity."""

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def perplexity(self, model: GPT2LMHeadModel, text: str) -> float:
        """
        Calculate perplexity of text under a model.
        Lower = model understands the text better.
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        return float(torch.exp(outputs.loss))

    def compare(self, original: GPT2LMHeadModel,
                finetuned: GPT2LMHeadModel,
                texts: list) -> dict:
        """
        Compare perplexity of two models on a list of texts.

        Returns:
            Dict with original and finetuned perplexities and percent change
        """
        orig_perps = [self.perplexity(original, t) for t in texts]
        ft_perps = [self.perplexity(finetuned, t) for t in texts]
        avg_orig = float(np.mean(orig_perps))
        avg_ft = float(np.mean(ft_perps))
        change = ((avg_ft - avg_orig) / avg_orig) * 100

        return {
            'original': avg_orig,
            'finetuned': avg_ft,
            'change_percent': change,
            'improved': change < 0
        }