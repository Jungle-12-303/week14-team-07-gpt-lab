# GPT 순전파와 역전파 이해하기

이 문서는 현재 프로젝트 코드 기준으로:

- 순전파가 어떤 순서로 흐르는지
- 역전파가 어떤 순서로 거꾸로 흐르는지
- 왜 gradient가 여러 갈래로 나뉘는지
- `TransformerBlock` 안에서 무엇이 먼저 계산되고, 무엇이 나중에 미분되는지

를 그림처럼 이해할 수 있게 설명합니다.

## 코드 위치 바로가기

- `TransformerBlock.forward` 구현: [src/model.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/src/model.py:111)
- `GPTModel.forward` 구현: [src/model.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/src/model.py:164)
- 한 배치 loss 계산: [src/train.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/src/train.py:13)
- 역전파 + 가중치 업데이트 루프: [src/train.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/src/train.py:159)
- 생성 함수: [src/train.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/src/train.py:89)
- 콘솔 생성 스크립트: [console_generate.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/console_generate.py:1)

---

## 1. 먼저 큰 그림

현재 모델의 큰 흐름은 이렇습니다.

```text
토큰 ID(idx)
   |
   v
InputEmbedding
   |
   v
TransformerBlock 1
   |
   v
TransformerBlock 2
   |
   v
...
   |
   v
final_norm
   |
   v
out_head
   |
   v
logits
   |
   v
cross_entropy
   |
   v
loss
```

이게 순전파입니다.

역전파는 이 화살표를 **정반대로** 따라갑니다.

```text
loss
   |
   v
cross_entropy
   |
   v
logits
   |
   v
out_head
   |
   v
final_norm
   |
   v
마지막 TransformerBlock
   |
   v
...
   |
   v
첫 번째 TransformerBlock
   |
   v
InputEmbedding
```

즉:

- 순전파: 앞에서 뒤로
- 역전파: 뒤에서 앞으로

입니다.

---

## 2. 현재 프로젝트 기준 순전파

`GPTModel.forward()` 안에서 대략 이런 코드 흐름이 있습니다.

```text
idx
 -> input_embedding(idx)
 -> block 1
 -> block 2
 -> ...
 -> final_norm
 -> out_head
 -> logits
 -> cross_entropy(targets가 있으면)
 -> loss
```

여기서 중요한 건:

- `input_embedding`은 정수 토큰 ID를 벡터로 바꿈
- `TransformerBlock`은 벡터를 문맥 반영된 더 좋은 벡터로 바꿈
- `final_norm`은 마지막 벡터를 정리함
- `out_head`는 그 벡터를 vocab 점수표(logits)로 바꿈
- `cross_entropy`는 정답과 비교해서 손실을 계산함

---

## 3. TransformerBlock 하나만 떼어 보기

현재 네가 선택한 방식은 `pre-norm`입니다.

즉 블록 하나 안의 순전파는 이런 구조입니다.

```text
입력 x
 |
 +------------------------------+
 |                              |
 v                              |
norm1                           |
 |                              |
 v                              |
attention                       |
 |                              |
 v                              |
attn_out                        |
 |                              |
 +-------- x + attn_out <-------+
              |
              v
              x2
              |
              +--------------------------+
              |                          |
              v                          |
              norm2                      |
              |                          |
              v                          |
              ffn                        |
              |                          |
              v                          |
              ffn_out                    |
              |                          |
              +----- x2 + ffn_out <------+
                        |
                        v
                      출력 out
```

이걸 수식처럼 쓰면:

```text
x1 = norm1(x)
attn_out = attention(x1)
x2 = x + attn_out

x3 = norm2(x2)
ffn_out = ffn(x3)
out = x2 + ffn_out
```

---

## 4. 역전파는 왜 "거꾸로" 흐르는가

손실 `loss`는 맨 마지막에 계산됩니다.

그래서 gradient는:

- 손실에 가장 가까운 연산부터 먼저 계산하고
- 점점 입력 쪽으로 거슬러 올라갑니다

즉 블록 하나의 역전파는 이 순서입니다.

```text
out = x2 + ffn_out
   ->
ffn_out = ffn(x3)
   ->
x3 = norm2(x2)
   ->
x2 = x + attn_out
   ->
attn_out = attention(x1)
   ->
x1 = norm1(x)
   ->
x
```

이걸 그림으로 다시 쓰면:

```text
출력 out
   |
   v
[x2 + ffn_out]
   |        \
   |         \
   v          v
  x2      ffn_out
            |
            v
           ffn
            |
            v
          norm2
            |
            v
            x2
            |
            v
      [x + attn_out]
        |        \
        |         \
        v          v
        x       attn_out
                   |
                   v
               attention
                   |
                   v
                 norm1
                   |
                   v
                    x
```

여기서 중요한 포인트:

- `x + attn_out`
- `x2 + ffn_out`

이 두 덧셈에서 gradient가 두 갈래로 나뉩니다.

이게 residual connection의 핵심입니다.

---

## 5. residual connection에서 gradient가 갈라지는 이유

예를 들어:

```text
y = a + b
```

라면 역전파 때 `y`의 gradient는:

- `a`로도 전달되고
- `b`로도 전달됩니다

즉:

```text
dy/da = 1
dy/db = 1
```

그래서 gradient가 이렇게 나뉘어요.

```text
      y
      |
      v
    [a+b]
    /   \
   v     v
  a       b
```

트랜스포머 블록 안에서는:

- 원래 입력 경로
- attention/ffn 경로

둘 다 gradient를 받습니다.

이게 깊은 모델에서도 학습이 잘 되게 도와줍니다.

---

## 6. attention 안에서는 무슨 일이 일어나나

`MultiHeadAttention` 안의 순전파 핵심은 이런 흐름입니다.

```text
x
 -> W_query(x) = Q
 -> W_key(x)   = K
 -> W_value(x) = V
 -> score = QK^T / sqrt(d)
 -> causal mask
 -> softmax
 -> attn_weights
 -> attn_weights @ V
 -> context
 -> out_proj(context)
 -> attention output
```

역전파는 반대입니다.

```text
attention output
 -> out_proj
 -> context
 -> attn_weights @ V
 -> softmax
 -> masked scores
 -> QK^T
 -> Q, K, V
 -> W_query, W_key, W_value
```

즉 attention 안에서도:

- 마지막 projection부터 gradient가 먼저 흐르고
- 점점 Q/K/V 쪽으로 거슬러 올라갑니다

---

## 7. `out_head`는 왜 필요한가

`final_norm`까지 끝난 `x`는 아직 그냥 hidden state입니다.

예를 들어 shape이:

```text
(batch_size, seq_len, emb_dim)
```

이라면 각 토큰 위치마다 `emb_dim` 크기의 벡터가 있는 상태입니다.

하지만 우리가 진짜 원하는 건:

```text
각 토큰 위치에서 vocab_size개 단어 후보의 점수
```

즉 shape이:

```text
(batch_size, seq_len, vocab_size)
```

인 logits예요.

그래서:

```text
hidden state --out_head--> logits
```

가 필요합니다.

---

## 8. cross entropy에서 왜 `view()`를 하냐

현재 코드에서는:

```python
loss = nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
)
```

를 씁니다.

원래 shape:

- `logits`: `(B, T, V)`
- `targets`: `(B, T)`

입니다.

그런데 cross entropy는 보통:

- 예측: `(N, C)`
- 정답: `(N,)`

형태를 기대합니다.

그래서:

- `B * T`개의 토큰 예측 문제를
- 하나의 긴 목록으로 펴서
- 한꺼번에 loss를 계산하는 거예요

시각적으로:

```text
(B, T, V)
  ->
(B*T, V)
```

```text
(B, T)
  ->
(B*T)
```

---

## 9. optimizer까지 포함한 전체 학습 흐름

실제 학습은 보통 이렇게 됩니다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

이걸 의미로 풀면:

### 1. `optimizer.zero_grad()`
- 이전 step에서 쌓인 gradient를 0으로 비움

### 2. `loss.backward()`
- loss부터 시작해서 그래프를 거꾸로 따라가며
- 모든 파라미터의 gradient를 계산

### 3. `optimizer.step()`
- 계산된 gradient를 사용해
- 가중치를 조금 수정

즉 학습 1 step은:

```text
순전파 -> loss 계산 -> 역전파 -> 가중치 업데이트
```

입니다.

---

## 10. 현재 프로젝트 기준 "역전파 순서" 한눈에 보기

### 순전파

```text
idx
 -> input_embedding
 -> block1
 -> block2
 -> ...
 -> final_norm
 -> out_head
 -> logits
 -> cross_entropy
 -> loss
```

### 역전파

```text
loss
 -> cross_entropy
 -> logits
 -> out_head
 -> final_norm
 -> 마지막 block
 -> ...
 -> 첫 번째 block
 -> input_embedding
```

### block 내부 역전파

```text
out = x2 + ffn_out
 -> ffn_out
 -> norm2
 -> x2 = x + attn_out
 -> attn_out
 -> attention
 -> norm1
 -> x
```

---

## 11. 정말 짧은 핵심 요약

### 순전파

- 입력을 점점 더 좋은 표현으로 바꾼다
- 마지막에 logits를 만들고 loss를 계산한다

### 역전파

- loss에서 시작해서 계산 그래프를 거꾸로 따라간다
- 각 연산은 자기 입력과 파라미터에 gradient를 나눠 준다
- residual에서는 gradient가 두 갈래로 나뉜다
- 마지막에 optimizer가 실제 가중치를 업데이트한다

---

## 12. 네가 딱 기억하면 좋은 문장

- `forward()`는 "값을 만드는 순서"
- `backward()`는 "그 값이 잘못된 책임을 거꾸로 나눠주는 순서"
- residual은 gradient가 우회로로도 흐르게 해준다
- pre-norm은 각 큰 연산 전에 입력을 안정화해 준다
- `out_head`는 hidden state를 vocab 점수표로 바꾼다
