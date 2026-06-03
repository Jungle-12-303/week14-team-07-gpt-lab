# mini GPT 자동 학습 실험 루프

이 폴더는 Codex 자동화가 반복적으로 가설을 세우고, 한 회차 실험을 실행하고, 결과를 분석한 뒤 다음 가설을 남기기 위한 작업 공간입니다.

## 현재 하드웨어 요약

```json
{
  "timestamp": "2026-06-03T04:14:28+00:00",
  "hostname": "woonyong-MacBookPro.local",
  "platform": "macOS-26.3.1-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "python": "3.13.13",
  "torch": "2.12.0",
  "cpu_count": 10,
  "memory_gb": 24.0,
  "cuda_available": false,
  "cuda_device_count": 0,
  "mps_available": true,
  "resolved_device": "mps",
  "profile": "mps_balanced"
}
```

## 실행 방식

자동화는 몇 분마다 다음 명령을 실행합니다.

```bash
python -m src.train_loop_agent
```

각 실행은 다음 순서로 동작합니다.

1. `docs/train/state.json`을 읽는다.
2. `docs/train/.train_loop.lock`이 살아 있으면 진행 중으로 보고 종료한다.
3. 직전 결과와 leaderboard를 분석한다.
4. Codex/LLM이 작성한 `next_plan.json`이 있으면 그 가설을 우선 사용하고, 없으면 규칙 기반 fallback으로 다음 가설과 설정을 만든다.
5. 한 회차 실험을 실행한다.
6. `docs/train/runs/run_XXX.md` 보고서를 쓴다.
7. `docs/train/leaderboard.csv`와 `docs/train/hypotheses.md`를 갱신한다.
8. `docs/train/dashboard.md`와 `docs/train/visuals/`의 시각 지표를 갱신한다.

## 결과 해석 기준

- `final_val_loss`가 낮을수록 좋다.
- `final_generalization_gap = final_val_loss - final_train_loss`가 커지면 과적합 위험이다.
- `overfit_score`는 낮을수록 좋다.
- `fit_status == "generalizing"`이면 다음에는 seed 반복으로 재현성을 확인한다.
- `fit_status == "overfit_risk"`이면 dropout, weight decay, tying, 모델 축소를 우선한다.

## 시각화

- `dashboard.md`: loss, generalization gap, overfit_score를 한 화면에서 보는 요약 대시보드
- `metrics_summary.csv`: 시각화와 해석에 쓰는 정규화된 지표 테이블
- `visuals/loss_overfit_trends.svg`: 모든 run의 train/val loss와 과적합 신호 추세
- `visuals/latest_run_metrics.svg`: 최신 run의 loss와 과적합 신호 막대 그래프
- `runs/run_XXX_artifacts/run_metrics.svg`: 각 회차 보고서에 포함되는 run별 상세 그래프

## 주요 파일

- `state.json`: 현재 상태와 best run
- `leaderboard.csv`: 모든 완료 실험 요약
- `metrics_summary.csv`: 주요 loss/과적합 지표 요약
- `dashboard.md`: 사람이 바로 볼 수 있는 시각 대시보드
- `visuals/`: SVG 그래프
- `hypotheses.md`: 가설과 결론 누적 기록
- `runs/run_XXX.md`: 회차별 보고서
- `runs/run_XXX_artifacts/`: 해당 실험의 plan/result 파일
- `next_plan.json`: Codex/LLM이 자유롭게 세운 다음 가설. 실행되면 해당 run artifact로 이동한다.
- `next_plan.schema.json`: `next_plan.json` 작성 형식
