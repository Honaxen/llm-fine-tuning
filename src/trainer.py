"""
trainer.py
----------
Fine-tuning trainer for TinyLlama using LoRA (PEFT).

Upgrade from v1:
  v1 — GPT-2 full fine-tuning (updates all parameters)
  v2 — TinyLlama-1.1B + LoRA (trains only ~4M of 1.1B parameters)

Why LoRA?
  Full fine-tuning GPT-2 (117M params) requires updating everything.
  LoRA on TinyLlama (1.1B params) trains 0.38% of weights — faster,
  less memory, and produces a tiny ~50MB adapter instead of a 2.2GB model.
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class LLMTrainer:
    """Fine-tune TinyLlama with LoRA on instruction data."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load base model and tokenizer."""
        print(f"Loading: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self._print_params()

    def _print_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"  Total    : {total:,}")

    def _tokenize(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        def tokenize_fn(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            result["labels"] = result["input_ids"].copy()
            return result
        return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def train(
        self,
        train_data: list[dict],
        eval_data: list[dict] = None,
        output_dir: str = "models/lora_adapter",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
    ) -> list[float]:
        """
        Fine-tune on instruction data.

        Args:
            train_data : list of {"text": "<|system|>...<|user|>...<|assistant|>..."}
            eval_data  : optional validation set
            output_dir : where to save the LoRA adapter
            epochs     : number of training epochs
            batch_size : per-device batch size
            learning_rate : AdamW learning rate
            max_seq_length : max tokens per example

        Returns:
            List of training losses per logging step
        """
        if self.model is None:
            self.load()

        train_dataset = self._tokenize(Dataset.from_list(train_data), max_seq_length)
        eval_dataset = self._tokenize(Dataset.from_list(eval_data), max_seq_length) if eval_data else None

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_total_limit=2,
            fp16=False,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        result = trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\nAdapter saved → {output_dir}")

        return [log["loss"] for log in trainer.state.log_history if "loss" in log]

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate a response for a formatted prompt."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()