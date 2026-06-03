# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다."""

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self._length = max(0, (len(token_ids) - context_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < self._length:
            raise IndexError(idx)

        start = idx * self.stride
        end = start + self.context_length
        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1 : end + 1]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GPTDataset(token_ids, context_length=context_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
