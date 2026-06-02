# 최소 실험 환경 가이드

이 프로젝트에는 최소한의 실험 스크립트로 다음 두 가지를 바로 실행할 수 있게 구성했습니다.

- 단일 학습 실행
- 하이퍼파라미터 스윕 실행

실행 파일:

- [run_experiment.py](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/run_experiment.py:1)

---

## 1. 왜 이 정도만 두었는가

코드를 너무 많이 늘리지 않기 위해:

- 기존 `src/bpe.py`, `src/dataset.py`, `src/model.py`, `src/train.py`를 최대한 재사용
- 실험 진입점만 한 파일로 제공
- 결과는 `json`과 `png`로 저장

하는 방식으로 최소화했습니다.

---

## 2. 실행 방법

### 단일 학습

```bash
python run_experiment.py train \
  --text-file data/nsmc_lm_train.txt \
  --output-dir artifacts/run1 \
  --num-epochs 3 \
  --vocab-size 300 \
  --context-length 32 \
  --emb-dim 32 \
  --n-heads 4 \
  --n-layers 1
```

생성물:

- `artifacts/run1/loss_curve.png`
- `artifacts/run1/train_result.json`

### 하이퍼파라미터 스윕

```bash
python run_experiment.py sweep \
  --text-file data/nsmc_lm_train.txt \
  --output-dir artifacts/sweep1 \
  --num-epochs 2
```

생성물:

- `artifacts/sweep1/sweep_results.json`
- `artifacts/sweep1/sweep_val_loss.png`

---

## 3. 성능 비교 기준

최소 환경에서 가장 중요한 1차 기준은 `validation loss`입니다.

### 1차 기준: `final_val_loss`

의미:

- 검증 데이터에서 다음 토큰을 얼마나 잘 맞추는지
- 낮을수록 일반화 성능이 좋을 가능성이 큼

목적:

- 학습 데이터에만 맞춘 모델이 아니라
- 실제로 덜 본 데이터에도 잘 맞는 설정을 찾기 위해 사용

### 2차 기준: `perplexity`

계산:

- `perplexity = exp(val_loss)`

의미:

- 모델이 다음 토큰을 얼마나 “덜 헷갈리는지”를 직관적으로 보는 값
- 낮을수록 좋음

목적:

- loss보다 좀 더 해석하기 쉬운 언어모델 품질 지표 제공

### 3차 기준: `train_loss`와 `val_loss` 차이

의미:

- `train_loss`는 낮은데 `val_loss`가 높으면 과적합 가능성이 큼

목적:

- 단순히 학습이 잘 된 것처럼 보이는 설정을 걸러내기 위해 사용

### 4차 기준: `num_params`

의미:

- 모델 크기

목적:

- 비슷한 성능이면 더 작은 모델을 선호할 수 있음

### 5차 기준: `elapsed_sec`

의미:

- 학습 시간

목적:

- 성능과 속도 사이의 균형을 보기 위해 사용

---

## 4. 어떤 파라미터를 먼저 바꿔볼까

최소 환경에서는 아래 순서를 추천합니다.

### `emb_dim`

의미:

- 토큰 표현 크기

기대 효과:

- 너무 작으면 표현력이 부족
- 너무 크면 느리고 과적합 가능성 증가

### `n_layers`

의미:

- 트랜스포머 블록 수

기대 효과:

- 깊어질수록 더 복잡한 문맥을 학습 가능
- 너무 깊으면 느려지고 학습이 어려워질 수 있음

### `drop_rate`

의미:

- dropout 강도

기대 효과:

- 과적합 완화
- 너무 크면 학습 자체가 약해질 수 있음

### `lr`

의미:

- 학습률

기대 효과:

- 너무 크면 불안정
- 너무 작으면 매우 느림

### `pre_norm` / `post_norm`

의미:

- 블록 내부 정규화 위치

기대 효과:

- `pre_norm`은 보통 더 안정적
- `post_norm`은 비교 실험용으로 의미 있음

---

## 5. 최적 파라미터를 고르는 간단한 규칙

1. `final_val_loss`가 가장 낮은 설정을 우선 후보로 둡니다.
2. 비슷하면 `perplexity`가 더 낮은 쪽을 봅니다.
3. 그래도 비슷하면 `num_params`가 더 적고 `elapsed_sec`가 더 짧은 쪽을 선택합니다.
4. `train_loss`는 계속 내려가는데 `val_loss`가 나빠지면 과적합으로 판단합니다.

---

## 6. 커스텀 스윕

기본 스윕 대신 직접 설정을 넣고 싶으면 JSON 파일로 리스트를 만들어 `--sweep-file`에 넘기면 됩니다.

예시:

```json
[
  {"emb_dim": 32, "n_layers": 1, "drop_rate": 0.0, "lr": 0.001},
  {"emb_dim": 64, "n_layers": 2, "drop_rate": 0.1, "lr": 0.0005}
]
```

실행:

```bash
python run_experiment.py sweep \
  --text-file data/nsmc_lm_train.txt \
  --output-dir artifacts/custom_sweep \
  --sweep-file my_sweep.json
```

