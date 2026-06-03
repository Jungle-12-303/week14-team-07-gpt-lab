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

"""
호출 흐름 요약

GPTModel.forward
 -> InputEmbedding.forward
 -> TransformerBlock.forward (n번 반복)
    -> LayerNorm.forward
    -> MultiHeadAttention.forward
    -> LayerNorm.forward
    -> FeedForward.forward
       -> GELU.forward
 -> LayerNorm.forward (final_norm)
 -> out_head
 -> optional: cross_entropy

"""


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""
    
    # 마지막 차원 크기: normalized_shape, 보정값: eps
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # 정규화된 각 feature 값을 다시 얼마나 크게/작게 쓸지 조절하는 학습 가능한 스케일 파라미터
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        # 정규화된 각 feature 값에 더해지는 학습 가능한 이동값(offset)
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps # 분산이 0에 가까울 때 나누기 오류를 막는 작은 값

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta

# GELU 활성화 함수를 tanh 근사식으로 계산해 반환
class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
                * (x + 0.044715 * x.pow(3))
            )
        )


class GELUExact(nn.Module):
    """정확한 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))


class QuickGELU(nn.Module):
    """Quick GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def build_activation(activation_name: str) -> nn.Module:
    """설정 문자열에 맞는 활성화 함수를 생성합니다."""
    if activation_name == "gelu":
        return GELU()
    if activation_name == "gelu_exact":
        return GELUExact()
    if activation_name == "quick_gelu":
        return QuickGELU()
    raise ValueError(f"Unsupported activation_name: {activation_name}")

# FFN : Feed-Forward Network
# 입력 x를 FFN 전체에 통과시켜 결과를 반환 <= 설명이 이게 다임?
class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4, activation_name: str = "gelu"):
        super().__init__()
        hidden_dim = mult * d_model
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            build_activation(activation_name),
            nn.Linear(hidden_dim, d_model), # 입력 차원을 더 큰 히든 차원으로 확장
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ FeedForward 네트워크를 통과시킵니다."""
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
        pre_norm: bool = True,
        activation_name: str = "gelu",
    ):
        super().__init__()
        self.pre_norm = pre_norm # pre-norm 방식인지 post-norm 방식인지 저장
        self.norm1 = LayerNorm(d_model) # attention 앞/뒤에 쓸 첫 번째 LayerNorm 생성
        self.attention = MultiHeadAttention( # 멀티헤드 어텐션 모듈 생성
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(d_model) # FFN 앞/뒤에 쓸 두 번째 LayerNorm 생성
        self.ffn = FeedForward(d_model=d_model, dropout=drop_rate, activation_name=activation_name) # FFN 모듈 생성

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """pre-norm/post-norm 설정에 따라 attention과 FFN을 residual로 연결합니다."""
        if self.pre_norm: # 입력 x를 먼저 정규화한 뒤 attention
            attn_out = self.attention(self.norm1(x), causal_mask=causal_mask)
            x = x + attn_out
            ffn_out = self.ffn(self.norm2(x)) # x를 FFN에 넣기
            x = x + ffn_out
            return x

        # 정규화 없이 attention을 먼저 수행 후, residual로 더한 뒤 정규화
        attn_out = self.attention(x, causal_mask=causal_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config # 설정 딕셔너리를 저장

        # embedding, blocks, final layernorm, lm_head를 정의하세요.
        # 설정값들을 변수로 꺼내기 / config 항목은 test_model.py의 GPT_CONFIG_SMALL를 참고
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]
        drop_rate = config["drop_rate"]
        n_heads = config["n_heads"]
        qkv_bias = config["qkv_bias"]
        n_layers = config["n_layers"]
        pre_norm = config.get("pre_norm", True)
        activation_name = config.get("activation_name", "gelu")
        
        # 토큰 ID를 입력 임베딩 벡터로 바꾸는 모듈 생성
        self.input_embedding = InputEmbedding(vocab_size, emb_dim,context_length, drop_rate)

        # TransformerBlock 을 n_layers개 생성 후 저장
        # nn.ModuleList: 파이토치에서 사용되는 서브 모듈들을 리스트 형태로 관리하는 클래스. 모듈 저장만. 자동실행X
        # nn.Sequential: 모듈 저장하고, x를 넣으면 순서대로 자동 호출
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    n_heads,
                    drop_rate,
                    qkv_bias,
                    pre_norm=pre_norm,
                    activation_name=activation_name,
                )
                for _ in range(n_layers) # 직접 호출
            ]
        )
    
        self.final_norm = LayerNorm(emb_dim) # 마지막 hidden state를 정규화할 LayerNorm로 만들기
        # 마지막 hidden state를 vocab 크기의 logits로 바꾸는 출력층
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)


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
        x = self.input_embedding(idx) # 입력 토큰 ID를 임베딩 벡터로

        # 각 트랜스포머 블록을 순서대로 통과시키며 x를 계속 업데이트
        for block in self.trf_blocks:
            x = block(x, causal_mask=True)

        x = self.final_norm(x) # 모든 블록을 지난 결과를 마지막으로 정규화
        logits = self.out_head(x) # 각 위치의 hidden state를 vocab 점수(logits)로 변환

        if targets is None: # 정답 토큰이 없으면 logits만 반환
            return logits

        # (batch, seq, vocab) 형태의 logits와 (batch, seq) 형태의 targets를 
        # 2차원/1차원으로 펴서 cross entropy loss를 계산
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return loss, logits # 학습용으로는 loss, 출력 해석/예측용으로는 logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
