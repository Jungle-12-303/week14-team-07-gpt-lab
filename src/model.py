# -*- coding: utf-8 -*-
"""GPT model components."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class GELU(nn.Module):
    """Approximate GELU with tanh, matching historical mini-GPT behavior."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class GELUExact(nn.Module):
    """Exact GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="none")


class QuickGELU(nn.Module):
    """Quick GELU used by CLIP-like models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SquaredReLU(nn.Module):
    """Squared ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)


def build_activation(activation_name: str) -> nn.Module:
    """Construct a supported FFN activation module."""

    if activation_name == "gelu":
        return GELU()
    if activation_name == "gelu_exact":
        return GELUExact()
    if activation_name == "quick_gelu":
        return QuickGELU()
    if activation_name == "silu":
        return nn.SiLU()
    if activation_name == "mish":
        return nn.Mish()
    if activation_name == "squared_relu":
        return SquaredReLU()
    if activation_name == "swiglu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation_name: {activation_name}")


class FeedForward(nn.Module):
    """Transformer FFN with historical activation/dropout variants."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        mult: int = 4,
        activation_name: str = "gelu",
        dropout_position: str = "after_output",
    ):
        super().__init__()
        if dropout_position not in {"after_output", "after_activation", "none"}:
            raise ValueError(f"Unsupported dropout_position: {dropout_position}")

        self.hidden_dim = mult * d_model
        self.activation_name = activation_name
        self.dropout_position = dropout_position
        self.activation = build_activation(activation_name)
        self.dropout = nn.Dropout(dropout) if dropout_position != "none" else None
        self.is_gated = activation_name == "swiglu"

        if self.is_gated:
            self.value_proj = nn.Linear(d_model, self.hidden_dim)
            self.gate_proj = nn.Linear(d_model, self.hidden_dim)
        else:
            self.in_proj = nn.Linear(d_model, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            hidden = self.value_proj(x) * self.activation(self.gate_proj(x))
        else:
            hidden = self.activation(self.in_proj(x))

        if self.dropout is not None and self.dropout_position == "after_activation":
            hidden = self.dropout(hidden)

        out = self.out_proj(hidden)

        if self.dropout is not None and self.dropout_position == "after_output":
            out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Attention + FFN block with configurable pre/post-norm behavior."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        pre_norm: bool = True,
        activation_name: str = "gelu",
        ffn_mult: int = 4,
        ffn_dropout_position: str = "after_output",
        attention_impl: str = "manual",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = LayerNorm(d_model, eps=norm_eps)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            attention_impl=attention_impl,
        )
        self.norm2 = LayerNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(
            d_model=d_model,
            dropout=drop_rate,
            mult=ffn_mult,
            activation_name=activation_name,
            dropout_position=ffn_dropout_position,
        )

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        if self.pre_norm:
            attn_out = self.attention(self.norm1(x), causal_mask=causal_mask)
            x = x + attn_out
            ffn_out = self.ffn(self.norm2(x))
            return x + ffn_out

        attn_out = self.attention(x, causal_mask=causal_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)

        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]
        drop_rate = config["drop_rate"]
        n_heads = config["n_heads"]
        qkv_bias = config["qkv_bias"]
        n_layers = config["n_layers"]
        pre_norm = config.get("pre_norm", config.get("norm_first", True))
        activation_name = config.get("activation_name", "gelu")
        ffn_mult = config.get("ffn_mult", 4)
        ffn_dropout_position = config.get("ffn_dropout_position", "after_output")
        attention_impl = config.get("attention_impl", "manual")
        tie_embeddings = config.get("tie_embeddings", False)
        init_std = config.get("init_std", 0.02)
        norm_eps = config.get("norm_eps", 1e-5)

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            context_length=context_length,
            drop_rate=drop_rate,
        )
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                    pre_norm=pre_norm,
                    activation_name=activation_name,
                    ffn_mult=ffn_mult,
                    ffn_dropout_position=ffn_dropout_position,
                    attention_impl=attention_impl,
                    norm_eps=norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(emb_dim, eps=norm_eps)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.tie_embeddings = tie_embeddings

        self._reset_parameters(init_std)
        if tie_embeddings:
            self.out_head.weight = self.input_embedding.token_embedding.weight

    def _reset_parameters(self, init_std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.gamma)
                nn.init.zeros_(module.beta)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_embedding(idx)
        for block in self.trf_blocks:
            x = block(x, causal_mask=True)
        x = self.final_norm(x)
        logits = self.out_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy text generation helper."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
    if was_training:
        model.train()
    return idx
