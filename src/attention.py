# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,   # 입력 벡터의 차원
        n_heads: int,   # attention head 개수
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model  # 전체 embedding 차원을 객체 안에 저장
        self.n_heads = n_heads  # head 개수를 저장
        self.head_dim = d_model // n_heads  # head 하나가 담당할 차원 수를 계산
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)   # 입력 x를 Query로 바꾸는 선형 변
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)     # 입력 x를 Key로 바꾸는 선형 변환
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)   # 입력 x를 Value로 바꾸는 선형 변환

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model) B=batch_size, T=seq_len, C=d_model
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, seq_len, d_model, = x.shape

        # 같은 입력 x를 세 관점의 벡터로 변환
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 현재 위치보다 미래 위치를 보지 못하게 막음
        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # attention 비율대로 Value 정보를 섞음
        context = attn_weights_dropped @ values

        # 나눴던 head를 다시 합쳐 입력과 같은 d_model 크기로 복원
        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, seq_len, d_model)

        out = self.out_proj(context)

        if return_attention_weights:
            return out, attn_weights
        return out