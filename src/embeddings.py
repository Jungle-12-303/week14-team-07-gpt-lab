# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        # nn.Embedding: 학습 가능한 lookup table
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, emb_dim) # 랜덤 vocab_size, emb_dim matrix 생성, 학습 가능한 파라미터
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate) # dropout network layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        seq_len = x.size(1) # x의 1번 차원의 길이 == seq_len
        positions = torch.arange(seq_len, device=x.device) # device = CPU / GPU. x와 position이 다른 연산장치에 있으면 서로에 대한 연산 불가
        x = self.token_embedding(x) + self.position_embedding(positions) # lookup
        return self.dropout(x)
