# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 유틸리티."""

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


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다."""

    def read_tsv(path: str | Path | None) -> list[dict]:
        if path is None:
            return []

        rows: list[dict] = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label is None:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    train_rows = read_tsv(train_tsv_path)
    test_rows = read_tsv(test_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(train_rows)
    val_size = int(len(train_rows) * val_ratio)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in (
            ("train", train_data),
            ("val", val_data),
            ("test", test_rows),
        ):
            with (output_path / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_data, val_data, test_rows


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

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
    """GPT backbone 위에 감성 분류용 Linear head를 붙인 모델."""

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

        non_pad_mask = input_ids != self.pad_id
        last_token_indices = non_pad_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_indices, last_token_indices]
        logits = self.classifier(self.dropout(pooled))

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels=labels)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples
