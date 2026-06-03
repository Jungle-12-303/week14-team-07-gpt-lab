# -*- coding: utf-8 -*-
"""Multi-head causal self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_score_matrix(
    queries: torch.Tensor,
    keys: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    scores = queries @ keys.transpose(-2, -1)
    if scale is not None:
        scores = scores / scale
    return scores


def apply_causal_score_mask(
    scores: torch.Tensor,
    seq_len: int | None = None,
) -> torch.Tensor:
    if seq_len is None:
        seq_len = scores.size(-1)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    return scores.masked_fill(mask, float("-inf"))


def normalize_attention_scores(
    scores: torch.Tensor,
    dropout: nn.Module | None = None,
) -> torch.Tensor:
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return attn_weights


def weighted_value_context(
    attn_weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return attn_weights @ values


class MultiHeadAttention(nn.Module):
    """GPT-style causal self-attention with manual/SDPA implementations."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        attention_impl: str = "manual",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attention_impl not in {"manual", "sdpa"}:
            raise ValueError(f"Unsupported attention_impl: {attention_impl}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_impl = attention_impl

        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("MultiHeadAttention input must have shape (B, T, C)")

        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        if self.attention_impl == "sdpa" and not return_attention_weights:
            context = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=causal_mask,
            )
            out = self._merge_heads(context)
            out = self.resid_dropout(self.out_proj(out))
            return out

        out, attn_weights = self._run_manual_attention(
            queries=queries,
            keys=keys,
            values=values,
            seq_len=seq_len,
            causal_mask=causal_mask,
        )
        if return_attention_weights:
            return out, attn_weights
        return out

    def _run_manual_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_len: int,
        causal_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = attention_score_matrix(queries, keys, scale=self.head_dim**0.5)
        if causal_mask:
            scores = apply_causal_score_mask(scores, seq_len)
        attn_weights = normalize_attention_scores(scores, dropout=self.attn_dropout)
        context = weighted_value_context(attn_weights, values)
        out = self._merge_heads(context)
        out = self.resid_dropout(self.out_proj(out))
        return out, attn_weights

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
