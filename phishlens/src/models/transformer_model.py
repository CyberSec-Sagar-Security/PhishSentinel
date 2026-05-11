"""
PhishLens DistilBERT Fine-Tuning Transformer Module.

Provides a DistilBERT-based sequence classifier that can be fine-tuned on
labelled phishing email data as an alternative to the feature-engineering
pipeline. The transformer model:
  - Takes raw email body text as input (no feature extraction required)
  - Produces a phishing probability from 0.0 to 1.0
  - Can be used standalone or as an ensemble member with the ML pipeline

Security rationale: Pre-trained language models capture subtle linguistic
patterns (tone, grammar, word choice) that are difficult to encode as
hand-crafted features. Fine-tuning DistilBERT on phishing emails bridges
the gap between the feature-engineering pipeline and end-to-end neural detection.

Architecture:
  - DistilBERT base uncased (66M parameters, 40% smaller than BERT-base)
  - Max sequence length: 512 tokens
  - Classification head: Linear(768) → Dropout(0.3) → Linear(2)
  - Training: AdamW with linear warmup, 3 epochs, batch size 16

Note: Requires GPU for practical training speed. CPU training is ~10x slower.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

# Detect GPU availability
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Transformer model using device: {DEVICE}")
except ImportError:
    DEVICE = "cpu"
    log.warning("PyTorch not found. DistilBERT fine-tuning unavailable.")


class DistilBERTPhishClassifier:
    """DistilBERT-based phishing email classifier.

    Args:
        model_name: HuggingFace model hub name (default: distilbert-base-uncased).
        max_length: Maximum tokeniser input length (default: 512).
        device: 'cuda' or 'cpu' (auto-detected if None).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or DEVICE
        self._tokenizer = None
        self._model = None
        self._is_fitted = False

    def __repr__(self) -> str:
        return (
            f"DistilBERTPhishClassifier("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"fitted={self._is_fitted})"
        )

    def _load_pretrained(self) -> None:
        """Load pre-trained DistilBERT tokeniser and model."""
        try:
            from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
            log.info(f"Loading {self.model_name} from HuggingFace hub ...")
            self._tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self._model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
            ).to(self.device)
            log.info("DistilBERT loaded successfully.")
        except ImportError:
            raise ImportError("transformers and torch required for DistilBERT fine-tuning.")

    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        val_split: float = 0.1,
        save_dir: Optional[str] = None,
    ) -> List[float]:
        """Fine-tune DistilBERT on labelled phishing email data.

        Args:
            texts: List of email body texts.
            labels: Binary labels (0=legitimate, 1=phishing).
            epochs: Number of training epochs (default: 3).
            batch_size: Training batch size (default: 16; reduce to 8 on CPU).
            learning_rate: AdamW learning rate.
            warmup_ratio: Fraction of training steps for linear LR warmup.
            val_split: Fraction of data for validation.
            save_dir: If provided, save fine-tuned model here.

        Returns:
            List of validation F1 scores per epoch.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset, random_split
        from transformers import AdamW, get_linear_schedule_with_warmup

        self._load_pretrained()

        class EmailDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx],
                    "labels": self.labels[idx],
                }

        log.info(f"Tokenising {len(texts):,} emails ...")
        dataset = EmailDataset(texts, labels, self._tokenizer, self.max_length)

        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        optimizer = AdamW(self._model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        val_f1_history = []
        from sklearn.metrics import f1_score

        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            self._model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(batch["labels"].numpy().tolist())

            val_f1 = float(f1_score(all_labels, all_preds, zero_division=0))
            val_f1_history.append(val_f1)
            log.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

        self._is_fitted = True

        if save_dir:
            self.save(save_dir)

        return val_f1_history

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict phishing probability for a list of email texts.

        Args:
            texts: List of email body texts.

        Returns:
            Array of shape [n, 2] with P(legitimate) and P(phishing) per row.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fine-tuned. Call fine_tune() first.")

        import torch
        import torch.nn.functional as F

        self._model.eval()
        all_probs = []

        for i in range(0, len(texts), 32):     # Process in batches of 32
            batch_texts = texts[i: i + 32]
            encodings = self._tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def save(self, save_dir: str) -> None:
        """Save the fine-tuned model and tokeniser to disk."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)
        log.info(f"DistilBERT model saved to '{save_dir}'")

    @classmethod
    def load(cls, load_dir: str) -> "DistilBERTPhishClassifier":
        """Load a fine-tuned model from disk."""
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        instance = cls(model_name=load_dir)
        instance._tokenizer = DistilBertTokenizer.from_pretrained(load_dir)
        instance._model = DistilBertForSequenceClassification.from_pretrained(load_dir).to(instance.device)
        instance._is_fitted = True
        log.info(f"DistilBERT model loaded from '{load_dir}'")
        return instance
