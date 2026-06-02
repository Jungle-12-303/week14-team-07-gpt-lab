# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn

# q와 K를 내적해서 attention score matrix를 만드는 함수
def attention_score_matrix(queries: torch.Tensor, keys: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    qk_dot_matrix = queries @ keys.transpose(-2, -1)
    if scale is not None:
        qk_dot_matrix = qk_dot_matrix / scale
    return qk_dot_matrix

# causal mask를 적용하는 함수 => 현재 위치보다 뒤에 있는 토큰의 score를 가려야 함
def apply_causal_score_mask(scores: torch.Tensor, seq_len:int | None = None) -> torch.Tensor :
    if seq_len is None:
        seq_len = scores.size(-1)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    return scores.masked_fill(mask, float("-inf"))

# score를 softmax로 정규화해서 attention weight로 만드는 함수
def normalize_attention_scores(scores: torch.Tensor, dropout: nn.Module | None = None) -> torch.Tensor:
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return attn_weights

def weighted_value_context(attn_weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return attn_weights @ values

# nn.Moudle: pytorch에서 제공하는 모든 모델의 근간이 되는 기본 클래스
# 모든 모델은 nn.Moudle을 상속받고 시작해야만 forward, backward 등을 편하게 수행 가능
class MultiHeadAttention(nn.Module): # # 여기서 nn.Module을 상속
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
        d_model: int, # 전체 임베딩 차원
        n_heads: int, # head 개수
        drop_rate: float = 0.1, # dropout 비율
        qkv_bias: bool = False, # Q/K/V 선형층에 bias를 넣을지 여부(T:파라미터를 내부적으로 생성)
    ):
        super().__init__() # nn.Module과 그 부모가 있다면 전부 상속
        
        if d_model % n_heads != 0: # 나눠떨어지지 않으면 head별 차원을 균등하게 나눌 수 없어서 에러
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads # 각 head가 담당할 차원

        # 입력 x에서 Query, Key, Value를 만들기 위한 선형층 3개
        # nn.Linear : pytorch에서 선형회귀 모델. in_features와 out_features를 매개변수로 받음.
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)

        # qkv projection, output projection, dropout을 정의
        # 모든 head 결과를 다시 합친 뒤 마지막으로 한 번 더 선형변환하는 출력 projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop_rate) # attention weight 쪽에 적용할 dropout
        self.resid_dropout = nn.Dropout(drop_rate) # 최종 출력 쪽에 적용할 dropout

        # raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요.")


    # 순전파
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

        if x.ndim != 3: # 입력 x가 3차원이 아니면 에러
            raise ValueError("MultiHeadAttention input must have shape (B, T, C)")
        
        seq_len = self._validate_embedding_dim(x)
        queries, keys, values = self._project_qkv(x) # 입력 x를 Query, Key, Value로 변환

        # 실제 어텐션 계산 : score 계산, mask 적용, softmax, V와 곱하기, head 합치기 등
        out, attn_weights = self._run_multi_head_attention(queries, keys, values, seq_len=seq_len, causal_mask=causal_mask)

        if return_attention_weights:
            return out, attn_weights
        return out # 최종 attention block


    # forward의 helper function
    def _validate_embedding_dim(self, x: torch.Tensor) -> int:
        # 입력의 마지막 차원이 모델 차원과 같은지 확인하고 seq_len 반환
        _, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")
        
        return seq_len



    def _project_qkv(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 입력 x에서 head별 Q, K, V를 만들기
        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        return queries, keys, values


    def _run_multi_head_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, seq_len: int, causal_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # score -> mask -> softmax -> context -> output projection 흐름 실행

        # queries와 keys로 attention score를 계산
        scores = attention_score_matrix(queries, keys, scale = self.head_dim**0.5)

        if causal_mask: # causal mask가 켜져 있으면 미래 토큰을 못 보게 가림
            scores = self._apply_causal_mask(scores, seq_len)

        # score를 softmax로 정규화, dropout 적용
        attn_weights = normalize_attention_scores(scores, dropout=self.attn_dropout)

        # attention weight와 value를 곱해서 context 생성, 여러 head 재병합.
        context = self._merge_head(weighted_value_context(attn_weights, values))

        return self.resid_dropout(self.out_proj(context)), attn_weights


    def _split_heads(self, x:torch.Tensor) -> torch.Tensor:
        # (B, T, C)를 (B, H, T, head_dim)로 교체
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)

        return x.transpose(1, 2)


    def _apply_causal_mask(self, scores: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        return apply_causal_score_mask(scores, seq_len)


    def _merge_head(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, head_dim)을 다시 (B, T, C)로 합체
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        
        return x.view(batch_size, seq_len, self.d_model)
