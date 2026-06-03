# train2

새 기준선과 새 가설을 기준으로 실험을 다시 쌓기 위한 별도 작업 공간입니다.

기본 원칙:

- 기존 `train/` 아카이브는 건드리지 않습니다.
- 앞으로의 새 실험 로그와 리포트는 `train/train2/` 아래에 쌓습니다.
- 대표 기준선은 [base_config.json](./base_config.json)을 출발점으로 수정해서 사용합니다.
- dense replay 기본 출력 경로는 `train/train2/replays`입니다.
- 새 `run_experiment.py train` 실행의 기본 출력 경로도 `train/train2/runs`입니다.

권장 구조:

- `runs/`: 회차별 실험 리포트/아티팩트
- `replays/`: plan 기반 dense replay 결과
- `visuals/`: 비교 그래프와 대시보드
- `hypotheses.md`: 가설 로그
- `leaderboard.csv`: 실험 요약 표
- `metrics_summary.csv`: 해석용 정리 표

주의:

- 기존 historical replay 스크립트의 기본 출력 경로도 이제 `train/train2/replays` 쪽을 기준으로 잡습니다.
- 새 실험을 시작할 때는 `base_config.json`을 복사/수정해서 run 계획의 기준점으로 삼는 걸 권장합니다.
- 새 replay를 돌리면 별도 옵션을 주지 않아도 결과가 `train/train2/replays/` 아래에 쌓입니다.
