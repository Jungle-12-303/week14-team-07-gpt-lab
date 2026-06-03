# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소."""

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
    """마지막 차원 기준 Layer Normalization."""

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
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
        )


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        hidden_dim = mult * d_model
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """GPT block: pre-norm causal self-attention and FFN with residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)
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
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

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

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    if was_training:
        model.train()
    return idx
