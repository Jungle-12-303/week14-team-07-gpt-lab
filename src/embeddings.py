# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn

# token ID를 의미를 담은 숫자 벡터로 바꾸는 것
# LLM은 raw text를 직접 처리하지 못하므로 text를 token ID로 바꾸고, 다시 embedding vector로 바꾼다
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
        vocab_size: int,    # tokenizer가 아는 token 개수
        emb_dim: int,   # token 하나를 몇 차원 벡터로 바꿀지
        context_length: int,    # 최대 몇 개 token 위치까지 처리할지
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)    # token embedding layer를 만들기
        self.position_embedding = nn.Embedding(context_length, emb_dim) # position embedding layer를 만들기. 몇 번째 위치인지를 나타내는 벡터
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.shape

        token_emb = self.token_embedding(x) # token ID를 embedding vector로 바꾸기

        positions = torch.arange(seq_len, device=x.device)  # device=x.device는 x가 CPU에 있으면 positions도 CPU에, GPU에 있으면 positions도 GPU에 만들라는 뜻
        pos_emb = self.position_embedding(positions)    # 위치 번호를 위치 벡터로 바꿈

        x = token_emb + pos_emb # token의 의미 정보와 token의 위치 정보를 합쳐서 Transformer에 넣을 최종 입력을 만든다
        x = self.dropout(x)

        return x
