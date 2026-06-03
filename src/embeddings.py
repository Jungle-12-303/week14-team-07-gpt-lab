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
        self.emb_dim = emb_dim
        self.context_length = context_length
        # token_embedding, position_embedding, dropout을 정의하세요.

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    # 입력으로 들어온 토큰 ID들에 대해 토큰 임베딩 + 위치 임베딩을 합쳐서, 트랜스포머가 먹을 수 있는 연속 벡터로 바꾸는 함수
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
         token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        if x.ndim != 2: # 잘못된 형식
            raise ValueError("token_ids must have shape (batch_size, seq_len)")
        _, seq_len = x.shape # _ : 값은 꺼내긴 하지만 안 쓸게요” 라는 관용적 표현

        # 너무 긴 입력이 들어와서 위치 임베딩 범위를 벗어나는 걸 미리 막는 검사
        if seq_len > self.context_length:
            raise ValueError(f"seq_len ({seq_len}) must be less than or equal to context_length ({self.context_length})"
        )
        
        token_embedding = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device) # 현재 입력 문장의 각 토큰 위치 번호 생성
        position_embedding = self.position_embedding(positions)

        return self.dropout(token_embedding + position_embedding)