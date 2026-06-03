# -*- coding: utf-8 -*-
"""NSMC sentiment classification fine-tuning utilities."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("document") or "").strip()
            label = row.get("label")
            if not text or label not in {"0", "1"}:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Read NSMC TSV files and return train/validation/test sentiment datasets.
    """
    train_rows = _read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    test_data = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("nsmc_sentiment_train.jsonl", train_data),
            ("nsmc_sentiment_val.jsonl", val_data),
            ("nsmc_sentiment_test.jsonl", test_data),
        ):
            with open(output_path / name, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """Sentiment classification Dataset returning one review and one label."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        input_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        input_ids = input_ids[: self.max_length]

        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.pad_id] * (self.max_length - len(input_ids))

        return torch.tensor(input_ids, dtype=torch.long), int(item["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone with a classification head for sentiment classification.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = pad_id
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x, causal_mask=True)
        x = self.gpt.final_norm(x)

        valid_lengths = (input_ids != self.pad_id).sum(dim=1)
        last_token_positions = torch.clamp(valid_lengths - 1, min=0)
        batch_positions = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_positions, last_token_positions]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the sentiment model for one epoch and return average loss/accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    if total == 0:
        return float("nan"), 0.0

    return total_loss / total, correct / total


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the sentiment model and return average loss/accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels=labels)
            total_loss += loss.item() * input_ids.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    model.train()

    if total == 0:
        return float("nan"), 0.0

    return total_loss / total, correct / total
