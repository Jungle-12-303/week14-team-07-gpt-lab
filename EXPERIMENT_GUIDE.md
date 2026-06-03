# 실험 가이드

이 브랜치의 실험 기준점은 `train/train2/`입니다.

핵심 원칙:

- 새 실험은 `run_experiment.py`로 실행합니다.
- 기본 출력 경로는 `train/train2/runs/`입니다.
- 과거 plan을 다시 돌리는 dense replay는 `train/replay_historical_run.py`를 사용합니다.
- 실험 결과는 최소 요약만이 아니라 `step/epoch/eval` 단위 로그까지 남깁니다.

---

## 1. 어디에 무엇이 쌓이나

새 실험:

- 기본 경로: `train/train2/runs/`
- 한 번 실행할 때마다 버전 디렉터리 1개 생성

historical replay:

- 기본 경로: `train/train2/replays/`

작업 공간:

- [train/train2/README.md](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/train/train2/README.md)
- [train/train2/base_config.json](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/train/train2/base_config.json)
- [train/train2/hypotheses.md](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/train/train2/hypotheses.md)

---

## 2. 가장 기본적인 실행

### 단일 실험

```bash
python run_experiment.py train \
  --text-file data/nsmc_lm_train.txt \
  --run-name baseline_a
```

이 명령은 별도 `--output-dir`를 주지 않으면 자동으로 `train/train2/runs/` 아래에 저장됩니다.

### historical replay

```bash
python train/replay_historical_run.py \
  --run-id 1 \
  --text-file data/nsmc_lm_train.txt
```

이 명령은 기본적으로 `train/train2/replays/` 아래에 저장됩니다.

### 하이퍼파라미터 스윕

```bash
python run_experiment.py sweep \
  --text-file data/nsmc_lm_train.txt \
  --output-dir train/train2/runs/sweeps \
  --num-epochs 2
```

---

## 3. 새 실험에서 조절 가능한 핵심 파라미터

기본 모델 크기/학습 설정:

- `vocab_size`
- `min_frequency`
- `context_length`
- `stride`
- `emb_dim`
- `n_heads`
- `n_layers`
- `drop_rate`
- `batch_size`
- `lr`
- `weight_decay`
- `grad_clip`
- `num_epochs`
- `eval_batches`
- `eval_every_steps`
- `train_ratio`
- `seed`

구조 실험용 설정:

- `activation_name`
  - `gelu`
  - `gelu_exact`
  - `quick_gelu`
  - `silu`
  - `mish`
  - `squared_relu`
  - `swiglu`
- `ffn_mult`
- `ffn_dropout_position`
  - `after_output`
  - `after_activation`
  - `none`
- `attention_impl`
  - `manual`
  - `sdpa`
- `tie_embeddings`
- `init_std`
- `norm_first`
- `norm_eps`

즉 지금 실험 경로는 단순 `emb_dim`, `n_layers`, `drop_rate` 정도만 만지는 최소 환경이 아니라, 모델 shape와 최적화 성질을 함께 바꿔보는 쪽까지 지원합니다.

---

## 4. 새 실험이 남기는 파일

실험 1회가 끝나면 보통 아래 파일들이 생깁니다.

요약/설정:

- `plan.json`
- `plan.csv`
- `results.jsonl`
- `results.csv`
- `train_result.json`
- `config.json`
- `tokenizer.json`

dense 로그:

- `step_history.csv`
- `step_history.jsonl`
- `eval_history.csv`
- `eval_history.jsonl`
- `epoch_history.csv`
- `epoch_history.jsonl`

곡선/시각화:

- `loss_curve.png`
- `perplexity_curve.png`
- `generalization_gap.png`
- `step_loss_curve.svg`
- `step_perplexity_curve.svg`
- `step_generalization_gap.svg`
- `epoch_loss_curve.svg`
- `epoch_perplexity_curve.svg`
- `epoch_generalization_gap.svg`
- `run_metrics.svg`

체크포인트:

- `best_checkpoint.pt`
- `final_checkpoint.pt`

추가 집계:

- `epoch_metrics.csv`
- 상위 출력 폴더의 `run_index.jsonl`

---

## 5. 로그에 실제로 남는 주요 컬럼

실험 설정 메타데이터:

- `learning_rate`
- `batch_size`
- `emb_dim`
- `dff`
- `n_heads`
- `n_layers`
- `ffn_mult`
- `activation_name`
- `ffn_dropout_position`
- `attention_impl`
- `tie_embeddings`
- `init_std`
- `norm_first`
- `parameter_count`
- `train_token_count`
- `val_token_count`

핵심 품질 지표:

- `initial_train_loss`
- `initial_val_loss`
- `final_train_loss`
- `final_val_loss`
- `final_train_perplexity`
- `final_val_perplexity`
- `initial_generalization_gap`
- `final_generalization_gap`
- `generalization_gap_delta`
- `train_loss_delta`
- `val_loss_delta`
- `train_val_improvement_gap`
- `overfit_score`
- `fit_status`
- `best_val_loss`
- `best_epoch`
- `best_step`
- `tokens_seen`
- `tokens_per_sec`
- `elapsed_sec`

즉 지금 로그 구조는:

- loss curve
- perplexity curve
- generalization gap curve
- lr x batch 성능 비교
- shape별 topology 비교
- `N = parameter_count`, `D = train_token_count` 기반 분석

까지 확장 가능한 형태입니다.

---

## 6. 어떤 지표를 기준으로 해석하나

### 1차 기준: `final_val_loss`

- 가장 직접적인 일반화 성능 기준
- 낮을수록 좋음

### 2차 기준: `final_val_perplexity`

- 언어모델 품질을 더 직관적으로 해석할 때 사용
- 낮을수록 좋음

### 3차 기준: `final_generalization_gap`

- `val_loss - train_loss`
- 크면 과적합 위험 증가 신호

### 4차 기준: `overfit_score`

- 현재 실험 경로에서는 과적합 신호를 한 숫자로 요약한 보조 지표
- 표준 metric은 아니고 자동 비교용 요약값
- 낮을수록 좋음

### 5차 기준: `fit_status`

- `generalizing`
- `overfit_risk`
- `underfit_or_too_short`

같은 해석 태그를 제공합니다.

### 6차 기준: `parameter_count`, `elapsed_sec`, `tokens_per_sec`

- 비슷한 품질이면 더 작은 모델, 더 빠른 설정이 유리할 수 있습니다.

---

## 7. 어떤 실험부터 시작하면 좋은가

### 기준선 확인

먼저 [train/train2/base_config.json](/home/leeminjeong/workspace/python_project/week14-team-07-gpt-lab/train/train2/base_config.json) 수준의 설정으로 기준선을 하나 잡습니다.

### 1차 축

- `learning_rate`
- `batch_size`
- `context_length`
- `stride`

이 축은 최적화 안정성과 일반화 차이를 빠르게 드러냅니다.

### 2차 축

- `emb_dim`
- `n_layers`
- `ffn_mult`
- `n_heads`

이 축은 모델 shape를 바꿉니다.

### 3차 축

- `activation_name`
- `attention_impl`
- `tie_embeddings`
- `ffn_dropout_position`
- `init_std`
- `norm_first`

이 축은 구조/학습 성질을 미세조정합니다.

---

## 8. 그래프를 만들 때 무엇을 전제로 하나

이 로그는 그래프를 그릴 수 있게 충분히 남지만, 그래프가 의미 있으려면 실험 설계가 그 축을 실제로 흔들어야 합니다.

예:

- `lr x batch_size` contour를 그리려면 `lr`와 `batch_size`를 여러 값으로 돌려야 함
- topology 비교를 하려면 `emb_dim`, `dff`, `n_layers`, `n_heads`를 바꿔가며 반복해야 함
- scaling law with `N`을 보려면 `parameter_count`가 달라지는 shape 실험이 필요함
- scaling law with `D`를 보려면 `train_token_count`가 달라지는 데이터 크기 실험이 필요함

즉 로그는 준비되어 있고, 이제 중요한 건 실험 매트릭스입니다.

---

## 9. 커스텀 스윕 예시

예시 `my_sweep.json`:

```json
[
  {
    "emb_dim": 64,
    "n_layers": 2,
    "ffn_mult": 3,
    "activation_name": "mish",
    "lr": 0.0003,
    "batch_size": 8
  },
  {
    "emb_dim": 96,
    "n_layers": 3,
    "ffn_mult": 4,
    "activation_name": "quick_gelu",
    "lr": 0.0002,
    "batch_size": 16
  }
]
```

실행:

```bash
python run_experiment.py sweep \
  --text-file data/nsmc_lm_train.txt \
  --output-dir train/train2/runs/custom_sweep \
  --sweep-file my_sweep.json
```

---

## 10. 지금 문서의 역할

이 문서는:

- 현재 브랜치에서 새 실험을 어떻게 돌리는지
- 어떤 로그가 남는지
- 어떤 지표로 해석하는지
- 어떤 분석 그래프를 만들 수 있는지

를 설명하는 **운영 가이드**입니다.

과거 `train/` 아카이브 해설 문서가 아니라, 앞으로 `train/train2/`에 실험을 쌓기 위한 기준 문서로 보면 됩니다.
