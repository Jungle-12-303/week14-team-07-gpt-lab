# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
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
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_tsv(path: str | Path | None) -> list[dict]:
        if path is None:
            return []
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
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
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", train_data), ("val", val_data), ("test", test_rows)):
            with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        ids = ids[: self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), int(item["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        batch_size, seq_len = input_ids.shape
        tok_emb = self.gpt.tok_emb(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.gpt.pos_emb(pos)
        x = self.gpt.drop_emb(tok_emb + pos_emb)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        pooled = x[:, -1, :]
        logits = self.classifier(self.dropout(pooled))
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits
        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
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

        total_loss += loss.item() * input_ids.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += input_ids.size(0)

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels=labels)

            total_loss += loss.item() * input_ids.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += input_ids.size(0)

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples
