"""
trainer.py
----------
Fine-tuning trainer for language models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class TextDataset(Dataset):
    """Dataset for causal language modeling."""

    def __init__(self, texts: list, tokenizer, max_length: int = 64):
        self.examples = []
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.examples.append(encoding.input_ids.squeeze())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {'input_ids': input_ids, 'labels': input_ids}


class Trainer:
    """Fine-tuning trainer for GPT-2 style models."""

    def __init__(self, model_name: str = 'distilgpt2',
                 device: str = None):
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    def train(self, texts: list, epochs: int = 10,
              learning_rate: float = 5e-5,
              batch_size: int = 4) -> list:
        """
        Fine-tune the model on a list of texts.

        Args:
            texts: List of training sentences
            epochs: Number of training epochs
            learning_rate: AdamW learning rate
            batch_size: Training batch size

        Returns:
            List of average losses per epoch
        """
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, labels=labels)
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
            losses.append(total_loss / len(dataloader))

        self.model.eval()
        return losses

    def save(self, path: str) -> None:
        """Save fine-tuned model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)