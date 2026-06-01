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
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model # embedding dimension
        self.n_heads = n_heads # number of heads
        self.head_dim = d_model // n_heads
        # qkv projection, output projection, dropout을 정의하세요.
        self.q_projection: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.k_projection: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.v_projection: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(drop_rate)
        # self.register_buffer(
        #     "mask",
        #     torch.triu(torch.ones(d_model, d_model), diagonal=1)
        # )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, seq_len, d_model = x.shape
        q: torch.Tensor = self.q_projection(x) # (batch_size, seq_len, d_model)
        k: torch.Tensor = self.k_projection(x)
        v: torch.Tensor = self.v_projection(x)

        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # (batch_size, seq_len, self.n_heads, self.head_dim) -> (batch_size, self.n_heads, seq_len, self.head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1) # t.transpose(-2, -1) == t.transpose(2, 3) # 마지막 두 차원을 사용한다는 의미이다.
        # '@'는 마지막 두 차원에 대해 행렬곱한다. 앞 차원들은 batch 차원으로 자동 정리
        """
        즉, 실제 의미는:
        for b in range(B):
            for h in range(H):
                result[b, h] = q[b, h] @ k_t[b, h]
        """

        # masking
        if causal_mask:
            mask_bool = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores.masked_fill_(mask_bool, -torch.inf) # inplace function. use masked_fill(...) for non-inplace.
        
        # softmax with scaling
        attn_weights = torch.softmax(
            attn_scores / k.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 문맥벡터 만들기.
        context_vec = (attn_weights @ v).transpose(1, 2)        

        context_vec = context_vec.contiguous().view(
            batch_size, seq_len, d_model
        )
        context_vec = self.out_projection(context_vec)
        if return_attention_weights:
            return context_vec, attn_weights
        else:
            return context_vec
