# -*- coding: utf-8 -*-
"""GPT model building blocks."""

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
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class GELU(nn.Module):
    """GPT-style GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class ReLU(nn.Module):
    """ReLU activation wrapper used for config-based FFN selection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class GELUExact(nn.Module):
    """Exact GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="none")


class QuickGELU(nn.Module):
    """Quick GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SquaredReLU(nn.Module):
    """Squared ReLU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)


def build_activation(activation_name: str) -> nn.Module:
    if activation_name == "gelu":
        return GELU()
    if activation_name == "gelu_exact":
        return GELUExact()
    if activation_name == "quick_gelu":
        return QuickGELU()
    if activation_name == "relu":
        return ReLU()
    if activation_name == "silu":
        return nn.SiLU()
    if activation_name == "mish":
        return nn.Mish()
    if activation_name == "squared_relu":
        return SquaredReLU()
    if activation_name == "swiglu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation_name}")


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> activation -> Linear -> Dropout."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        mult: int = 4,
        activation: str | None = None,
        activation_name: str | None = None,
        dropout_position: str = "after_output",
    ):
        super().__init__()
        if dropout_position not in {"after_output", "after_activation", "none"}:
            raise ValueError(f"Unsupported dropout_position: {dropout_position}")

        self.hidden_dim = mult * d_model
        self.activation_name = activation_name or activation or "gelu"
        self.dropout_position = dropout_position
        self.activation = build_activation(self.activation_name)
        self.dropout = nn.Dropout(dropout) if dropout_position != "none" else None
        self.is_gated = self.activation_name == "swiglu"

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
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        activation: str = "gelu",
        activation_name: str | None = None,
        ffn_mult: int = 4,
        ffn_dropout_position: str = "after_output",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ln2 = LayerNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(
            d_model,
            dropout=drop_rate,
            mult=ffn_mult,
            activation=activation,
            activation_name=activation_name,
            dropout_position=ffn_dropout_position,
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), causal_mask=causal_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        emb_dim = config["emb_dim"]
        vocab_size = config["vocab_size"]
        ffn_mult = config.get("ffn_mult", 4)
        ffn_dropout_position = config.get("ffn_dropout_position", "after_output")
        norm_eps = config.get("norm_eps", 1e-5)
        activation_name = config.get("activation_name", config.get("activation", "gelu"))

        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )
        self.input_embedding = self.embedding
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                    activation_name=activation_name,
                    ffn_mult=ffn_mult,
                    ffn_dropout_position=ffn_dropout_position,
                    norm_eps=norm_eps,
                )
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = LayerNorm(emb_dim, eps=norm_eps)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        if config.get("tie_embeddings", False):
            self.lm_head.weight = self.embedding.token_embedding.weight
        self.out_head = self.lm_head

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x, causal_mask=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy decoding으로 max_new_tokens만큼 다음 토큰을 생성합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
