# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

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
        """마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var+self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return nn.functional.gelu(x)


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)


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
    ):
        super().__init__()
        # attention, ffn, layernorm, dropout을 정의하세요.
        self.att = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.ff = FeedForward(d_model, dropout=drop_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, causal_mask=causal_mask)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.tok_emb = nn.Embedding(self.config['vocab_size'], self.config['emb_dim'])
        self.pos_emb = nn.Embedding(self.config['context_length'], self.config['emb_dim'])
        self.drop_emb = nn.Dropout(self.config['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                self.config['emb_dim'], 
                self.config['n_heads'],
                drop_rate=self.drop_emb,
                qkv_bias=self.config['qkv_bias']) for _ in range(self.config['n_layers'])]
        )

        self.final_norm = LayerNorm(self.config['emb_dim'])
        self.out_head = nn.Linear(self.config['emb_dim'], self.config['vocab_size'], bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        batch_size, seq_len = idx.shape
        tok_embds = self.tok_emb(idx)

        pos_embds = self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = tok_embds + pos_embds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        if targets is not None:
            # logits: (B, T, vocab_size) // targets: (B, T) // T == seq_len == n_token
            # logtis.view(-1, logits.size(-1)) = (B, T, vocab_size) == (B * T, vocab_size)
            # targts.view(-1) -> (B, T) -> (B * T)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
            )
            return loss, logits
        return logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,  # (batch_size, start_len = seq_len)
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # 마지막 context_size만 잘라냄.
        with torch.no_grad(): # 추론할 때에는 gradient 연산 불필요.
            logits = model(idx_cond) # GPT모델의 추론 값 = Logits
        
        logits = logits[:, -1, :] # logits에 들어있는 마지막 위치의 예측만 가져옴. (batch_size, token_len, vocb_size) -> (B, vocab_size)
        probas = torch.softmax(logits, dim=-1) # 마지막 차원 기준으로 softmax
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # 확률이 가장 높은 토큰 ID 가져오기
        idx = torch.cat((idx, idx_next), dim=1) # concat [origin_tokens, new_tokens]
    return idx
