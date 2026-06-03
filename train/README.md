# train workspace

이 브랜치의 `train/`은 **새 실험을 위한 최소 작업공간**만 남깁니다.

핵심 원칙:

- 과거 실험 아카이브와 리포트는 포함하지 않습니다.
- 앞으로의 실험 로그와 아티팩트는 `train/train2/` 아래에 쌓습니다.
- 기본 기준선은 [train2/base_config.json](./train2/base_config.json)에서 시작합니다.

## 주요 경로

- [train2/README.md](./train2/README.md): 새 실험 작업공간 안내
- [train2/base_config.json](./train2/base_config.json): 새 기준선 설정
- [train2/hypotheses.md](./train2/hypotheses.md): 앞으로의 가설 기록
- `train2/runs/`: 새 실험 리포트/요약 저장 위치
- `train2/replays/`: dense replay 산출물 저장 위치
- `train2/visuals/`: 새 비교 그래프 저장 위치

## Historical Replay

historical `plan.json`을 현재 코드로 다시 실행하면서 step/epoch 단위 raw 로그를 남기려면 아래 스크립트를 사용합니다.

```bash
python train/replay_historical_run.py \
  --run-id 21 \
  --text-file data/nsmc_lm_train_small.txt
```

기본 출력 경로는 `train/train2/replays/`입니다.

생성되는 주요 산출물:

- `step_history.csv/jsonl`
- `eval_history.csv/jsonl`
- `epoch_history.csv/jsonl`
- `step_loss_curve.svg`
- `step_perplexity_curve.svg`
- `step_generalization_gap.svg`
- `epoch_loss_curve.svg`
- `epoch_perplexity_curve.svg`
- `epoch_generalization_gap.svg`
- `run_metrics.svg`
- `checkpoint.pt`
