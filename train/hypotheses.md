# 자동 실험 가설 기록

아직 완료된 실험이 없습니다.

## run 001 - 2026-06-02T18:53:52+00:00

- 보고서: `docs/train/runs/run_001.md`
- 이번 가설: 기준선 수립: 현재 하드웨어에 맞춘 작은 GPT 설정으로 train/val gap과 처리량 기준점을 만든다.
- 근거: 아직 완료된 실험이 없으므로 과적합 판단과 속도 비교의 기준점이 필요하다.
- 바꾼 변수: `{"activation_name": "gelu", "attention_impl": "manual", "batch_size": 4, "context_length": 48, "drop_rate": 0.1, "emb_dim": 96, "eval_batches": 4, "ffn_dropout_position": "after_output", "ffn_mult": 4, "grad_clip": 1.0, "hypothesis": "기준선 수립: 현재 하드웨어에 맞춘 작은 GPT 설정으로 train/val gap과 처리량 기준점을 만든다.", "init_std": 0.02, "learning_rate": 0.0003, "max_steps": 20, "min_frequency": 2, "n_heads": 4, "n_layers": 2, "norm_eps": 1e-05, "norm_first": false, "qkv_bias": false, "run_id": 1, "seed": 123, "stride": null, "tie_embeddings": false, "train_ratio": 0.9, "vocab_size": 600, "weight_decay": 0.01}`
- 기대 결과: validation loss가 소폭 내려가고 generalization gap이 작게 유지된다.
- 실제 결과: final_val_loss=6.421781539916992, gap=0.008163809776306152, overfit_score=0.009853482246398926, fit_status=underfit_or_too_short
- 과적합 판단: 과소학습 또는 너무 짧은 학습. final gap=0.0082. step/capacity 증가를 검토한다.
- 다음 가설: 성공 시 같은 계열에서 seed 반복 또는 약한 capacity 증가를 검토한다. / 과적합 시 dropout, weight_decay, tie_embeddings를 강화한다.

## run 002 - 2026-06-02T19:00:52+00:00

- 보고서: `docs/train/runs/run_002.md`
- 이번 가설: MPS 기준선 재수립: run 001은 max_steps=1 smoke라 과소학습 신호가 강했으므로, Apple Silicon MPS에 맞춘 중간 크기 기준선으로 validation 개선과 gap 안정성을 다시 측정한다.
- 근거: leaderboard의 유일한 완료 실험은 final_val_loss=6.4218, final_generalization_gap=0.0082, overfit_score=0.0099, fit_status=underfit_or_too_short였다. 이 결과는 모델 구조나 regularization의 한계라기보다 1 step smoke 실행의 영향이 크다. 현재 하드웨어는 mps_balanced로 감지되었으므로 batch_size, context_length, emb_dim을 조금 올리고 max_steps를 40으로 늘려 과적합 이전의 실제 학습 곡선을 확인하는 것이 우선이다.
- 바꾼 변수: `{"batch_size": 8, "context_length": 64, "emb_dim": 128, "learning_rate": 0.0003, "max_steps": 40, "n_layers": 2, "seed": 134}`
- 기대 결과: train/val loss가 함께 더 뚜렷하게 내려가고, final_generalization_gap은 0.05 이하로 유지된다. fit_status가 generalizing 또는 mixed로 바뀌면 이후 seed 반복 또는 activation 교체를 검토한다.
- 실제 결과: final_val_loss=5.763571500778198, gap=0.053310513496398926, overfit_score=0.17463958263397217, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0533, overfit_score=0.1746. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 같은 설정을 seed만 바꿔 재현성을 확인하거나 quick_gelu/silu처럼 구조를 거의 바꾸지 않는 activation 교체를 한 축씩 실험한다. / 과적합 시 drop_rate를 0.15~0.2로 올리고 weight_decay를 0.05까지 강화하거나 tie_embeddings=True를 적용해 capacity 대비 일반화를 확인한다.

## run 003 - 2026-06-02T19:07:05+00:00

- 보고서: `docs/train/runs/run_003.md`
- 이번 가설: 정규화 강도 단일축 테스트: run 002는 validation loss를 크게 낮췄지만 final_generalization_gap과 overfit_score가 함께 커졌으므로, 같은 seed와 같은 MPS 기준선에서 drop_rate만 0.10에서 0.20으로 올리면 validation 성능을 크게 잃지 않으면서 gap을 줄일 수 있다.
- 근거: run 002는 final_val_loss=5.7636으로 run 001보다 좋아졌지만 final_generalization_gap=0.0533, generalization_gap_delta=0.0607, train_val_improvement_gap=0.0607, overfit_score=0.1746으로 train 쪽 개선이 validation보다 앞섰다. parameter_count=557824와 tokens_per_sec=8252는 현재 MPS 장비에서 부담 없는 범위이므로 capacity를 바로 줄이기보다 seed=134, max_steps=40, batch_size=8, context_length=64, emb_dim=128, n_layers=2를 고정하고 dropout만 강화해 과적합 신호가 완화되는지 분리해서 본다.
- 바꾼 변수: `{"drop_rate": 0.2}`
- 기대 결과: final_val_loss가 run 002 대비 소폭 상승하더라도 final_generalization_gap은 0.04 이하로 낮아지고 overfit_score는 0.12 이하로 내려간다. gap이 줄면서 validation 손실이 유지되면 dropout 강화가 현재 작은 말뭉치에서 유효한 정규화 축이라는 근거가 된다.
- 실제 결과: final_val_loss=5.774217367172241, gap=0.05174839496612549, overfit_score=0.16995322704315186, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0517, overfit_score=0.1700. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 drop_rate=0.20 설정을 유지하고 seed만 바꿔 재현성을 확인한다. 재현되면 quick_gelu 또는 silu처럼 구조를 바꾸지 않는 activation 단일축 실험으로 이동한다. / 과적합 시 dropout만으로 gap이 충분히 줄지 않으면 weight_decay를 0.05로 올리거나 tie_embeddings=True를 적용해 parameter_count와 overfit_score를 함께 비교한다.

## run 004 - 2026-06-02T19:16:07+00:00

- 보고서: `docs/train/runs/run_004.md`
- 이번 가설: 입출력 임베딩 공유 단일축 테스트: run 002가 현재 best지만 overfit_risk이고, run 003의 dropout 강화는 gap을 아주 조금만 낮췄다. 따라서 run 002 설정을 거의 고정하고 tie_embeddings=True만 적용하면 parameter_count가 줄어 작은 말뭉치에서 외우는 힘이 낮아져 final_generalization_gap과 overfit_score가 개선될 수 있다.
- 근거: dashboard 기준 최신 run 003은 final_val_loss=5.7742, final_generalization_gap=0.0517, overfit_score=0.1700으로 run 002보다 validation은 조금 나빠졌고 과적합 신호도 기대만큼 줄지 않았다. run 002는 final_val_loss=5.7636으로 best 후보지만 gap=0.0533, overfit_score=0.1746이 높다. dropout만으로는 충분하지 않았으므로 학습 길이, seed, 모델 폭, 깊이, learning_rate, dropout을 run 002와 동일하게 유지하고 weight tying만 적용해 capacity regularization의 순수 효과를 확인한다.
- 바꾼 변수: `{"tie_embeddings": true}`
- 기대 결과: parameter_count가 감소하고 final_val_loss는 run 002와 비슷한 5.76~5.82 범위에 머물면서 final_generalization_gap은 0.045 이하, overfit_score는 0.14 이하로 내려간다. validation 손실이 크게 악화되지 않으면 tie_embeddings는 작은 데이터에서 유효한 regularization 후보로 본다.
- 실제 결과: final_val_loss=5.755529403686523, gap=0.036219239234924316, overfit_score=0.17341935634613037, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0362, overfit_score=0.1734. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 tie_embeddings=True를 유지하고 seed만 바꿔 재현성을 확인한다. 재현되면 activation_name=quick_gelu 또는 silu 같은 함수 교체를 단일축으로 비교한다. / 과적합 시 tie_embeddings만으로도 과적합이 남으면 weight_decay=0.05를 단일축으로 올리거나 n_layers/ffn_mult를 줄이는 capacity 축소 실험으로 이동한다.

## run 005 - 2026-06-02T19:18:21+00:00

- 보고서: `docs/train/runs/run_005.md`
- 이번 가설: weight_decay 단일축 정규화 테스트: run 004는 tie_embeddings=True로 validation loss와 final gap을 개선해 best가 되었지만 overfit_score는 여전히 높다. run 004 설정을 유지하고 weight_decay만 0.01에서 0.05로 올리면 train 쪽 과도한 개선을 누그러뜨려 overfit_score와 train_val_improvement_gap을 낮출 수 있다.
- 근거: run 004는 final_val_loss=5.7555로 현재 best이며 final_generalization_gap도 0.0362까지 낮아졌다. 하지만 generalization_gap_delta=0.0686, train_val_improvement_gap=0.0686, overfit_score=0.1734로 fit_status는 계속 overfit_risk다. dropout 단일축(run 003)은 효과가 작았고 tie_embeddings(run 004)는 loss/gap 개선을 만들었으므로, 같은 tie_embeddings=True 설정을 유지한 채 weight_decay만 강화해 parameter_count와 구조를 그대로 두고 최적화 regularization이 과적합 신호를 완화하는지 분리해서 본다.
- 바꾼 변수: `{"weight_decay": 0.05}`
- 기대 결과: final_val_loss가 run 004 대비 크게 악화되지 않는 5.75~5.82 범위에 머물고, train_val_improvement_gap과 overfit_score가 줄어든다. 특히 final_generalization_gap은 0.04 이하를 유지하고 overfit_score는 0.15 이하로 낮아지는 것을 기대한다.
- 실제 결과: final_val_loss=5.755751132965088, gap=0.03619062900543213, overfit_score=0.1733335256576538, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0362, overfit_score=0.1733. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 weight_decay=0.05와 tie_embeddings=True 조합을 seed만 바꿔 재현성 검증한다. 재현되면 activation_name=quick_gelu 또는 silu 단일축 실험으로 넘어간다. / 과적합 시 weight_decay 강화 후에도 overfit_score가 높으면 n_layers=1 또는 ffn_mult=3처럼 capacity를 한 축씩 줄여 parameter_count와 validation 안정성을 함께 비교한다.

## run 006 - 2026-06-02T19:23:28+00:00

- 보고서: `docs/train/runs/run_006.md`
- 이번 가설: FFN 용량 축소 단일축 테스트: run 004의 tie_embeddings=True 설정은 validation loss와 final gap을 개선했지만 overfit_score는 여전히 높다. weight_decay 강화(run 005)는 거의 효과가 없었으므로, run 004 설정을 유지하고 ffn_mult만 4에서 3으로 줄이면 token별 FFN 암기 용량이 낮아져 validation 성능을 크게 잃지 않으면서 overfit_score와 train_val_improvement_gap이 완화될 수 있다.
- 근거: leaderboard와 dashboard에서 run 004는 final_val_loss=5.7555, final_generalization_gap=0.0362로 best 후보지만 overfit_score=0.1734와 fit_status=overfit_risk가 남아 있다. run 005는 weight_decay=0.05를 적용했지만 final_val_loss=5.7558, gap=0.0362, overfit_score=0.1733으로 사실상 변화가 없었다. 따라서 최적화 regularization보다 모델의 FFN capacity가 작은 말뭉치에서 train 개선 편향을 만드는지 확인할 차례다. n_layers를 바로 줄이면 attention/FFN block 전체 깊이가 바뀌므로, 더 작은 변화인 ffn_mult=3을 먼저 단일축으로 테스트한다.
- 바꾼 변수: `{"ffn_mult": 3}`
- 기대 결과: parameter_count가 run 004보다 줄고 final_val_loss는 5.75~5.85 범위에 머문다. 성공 기준은 final_generalization_gap이 0.04 이하를 유지하면서 train_val_improvement_gap 또는 overfit_score가 run 004보다 의미 있게 낮아지는 것이다.
- 실제 결과: final_val_loss=5.776598215103149, gap=0.052165985107421875, overfit_score=0.16454029083251953, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0522, overfit_score=0.1645. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 ffn_mult=3과 tie_embeddings=True 조합을 seed만 바꿔 재현성 검증한다. 재현되면 activation_name=quick_gelu 또는 silu를 단일축으로 비교한다. / 과적합 시 ffn_mult 축소 후에도 overfit_score가 높거나 validation이 악화되면 n_layers=1 단일축 실험으로 깊이 자체를 줄여본다.

## run 007 - 2026-06-02T19:28:34+00:00

- 보고서: `docs/train/runs/run_007.md`
- 이번 가설: best 설정 seed 재현성 검증: run 004의 tie_embeddings=True 설정은 현재까지 가장 낮은 final_val_loss와 안정적인 final gap을 보였지만, 아직 seed=134 한 번의 결과다. 구조나 regularization 축을 더 바꾸기 전에 seed만 151로 바꿔도 validation 개선과 gap 완화가 유지되는지 확인한다.
- 근거: run 004는 final_val_loss=5.7555, final_generalization_gap=0.0362로 현재 best다. run 005의 weight_decay=0.05는 거의 변화가 없었고, run 006의 ffn_mult=3은 parameter_count와 속도는 개선했지만 final_val_loss=5.7766, final_generalization_gap=0.0522로 악화되었다. 따라서 run 004의 개선은 tie_embeddings 자체의 효과일 수도 있고 seed=134 초기화/셔플의 우연일 수도 있다. 현재 하드웨어는 mps_balanced이고 run 시간이 1초 내외로 짧으므로, seed 반복 검증은 비용이 낮고 다음 activation/capacity 실험의 기준 신뢰도를 높인다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: final_val_loss가 5.75~5.85 범위에 머물고 final_generalization_gap이 0.05 이하로 유지되면 run 004 계열이 seed 하나의 우연이 아닐 가능성이 커진다. overfit_score는 아직 높을 수 있지만 run 006보다 validation/gap이 안정적이면 tie_embeddings=True를 기준선으로 채택한다.
- 실제 결과: final_val_loss=5.754852533340454, gap=0.04696786403656006, overfit_score=0.13947045803070068, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0470, overfit_score=0.1395. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 tie_embeddings=True 기준선을 유지하고 activation_name=quick_gelu 또는 silu를 단일축으로 비교한다. seed 반복 둘 다 안정적이면 활성함수 실험의 해석 신뢰도가 올라간다. / 과적합 시 seed 변경에서 validation이 크게 흔들리거나 gap이 커지면 seed variance가 큰 것으로 보고 run 004 계열을 2~3개 seed로 더 반복한 뒤 평균 기준으로 판단한다.

## run 008 - 2026-06-02T19:33:17+00:00

- 보고서: `docs/train/runs/run_008.md`
- 이번 가설: quick_gelu 활성함수 단일축 테스트: tie_embeddings=True 기준선은 seed=151에서 generalizing을 만들었으므로, 같은 설정에서 activation_name만 gelu에서 quick_gelu로 바꾸면 GELU 근사 특성으로 validation 성능을 유지하면서 처리량 또는 overfit_score가 개선될 수 있다.
- 근거: run 007은 seed=151, tie_embeddings=True, gelu 설정에서 final_val_loss=5.7549, final_generalization_gap=0.0470, overfit_score=0.1395, fit_status=generalizing으로 현재 best다. run 004와 run 007을 함께 보면 tie_embeddings=True 계열은 seed를 바꿔도 validation 성능이 유지된다. weight_decay 강화(run 005)는 변화가 거의 없었고 ffn_mult 축소(run 006)는 validation/gap을 악화시켰으므로, capacity 축소보다 작은 함수 교체가 다음에 더 안전한 실험 축이다. quick_gelu는 구조와 parameter_count를 바꾸지 않는 GELU 근사 계열이므로 현재 기준선과 의미 있게 비교하기 쉽다.
- 바꾼 변수: `{"activation_name": "quick_gelu"}`
- 기대 결과: final_val_loss가 run 007과 비슷한 5.74~5.82 범위에 머물고 final_generalization_gap이 0.05 이하로 유지된다. tokens_per_sec가 올라가거나 overfit_score가 0.139 이하로 낮아지면 quick_gelu를 유망한 GELU 대체 후보로 본다.
- 실제 결과: final_val_loss=5.75455904006958, gap=0.04693174362182617, overfit_score=0.13937902450561523, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0469, overfit_score=0.1394. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 quick_gelu가 안정적이면 같은 activation을 다른 seed로 반복해 재현성을 확인한다. 이후 silu 또는 gelu_exact와 비교해 함수 계열별 의미를 정리한다. / 과적합 시 quick_gelu에서 gap이나 overfit_score가 커지면 gelu 기준선으로 되돌리고 silu처럼 부드러운 activation 대안을 단일축으로 테스트한다.

## run 009 - 2026-06-02T19:38:28+00:00

- 보고서: `docs/train/runs/run_009.md`
- 이번 가설: quick_gelu seed 재현성 검증: run 008은 seed=151에서 quick_gelu가 gelu 기준선(run 007)보다 validation loss, overfit_score, 처리량을 모두 조금 개선했다. 같은 quick_gelu + tie_embeddings=True 설정을 seed=134에서도 실행하면 run 004(gelu, seed=134) 대비 generalization gap과 overfit_score가 유지 또는 개선되는지 확인할 수 있다.
- 근거: 현재 best는 run 008로 final_val_loss=5.7546, final_generalization_gap=0.0469, overfit_score=0.1394, fit_status=generalizing이다. 하지만 quick_gelu의 직접 비교는 seed=151에서 run 007 gelu와만 이루어졌다. seed=134에서는 gelu + tie_embeddings=True인 run 004가 final_val_loss=5.7555, gap=0.0362였지만 overfit_score=0.1734로 높았다. quick_gelu를 seed=134에도 적용하면 activation 교체가 특정 seed의 우연인지, 아니면 tie_embeddings=True 기준선 전반에서 유효한지 더 잘 판단할 수 있다. 실행 시간이 짧고 MPS에서 부담이 낮으므로 seed 재현성 검증의 정보 가치가 높다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: final_val_loss가 5.75~5.82 범위에 머물고 final_generalization_gap이 0.05 이하로 유지된다. run 004보다 overfit_score가 낮거나 run 008과 같은 generalizing 판정이 나오면 quick_gelu를 seed-robust 후보로 본다.
- 실제 결과: final_val_loss=5.75526762008667, gap=0.03619587421417236, overfit_score=0.17334282398223877, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0362, overfit_score=0.1733. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 quick_gelu가 seed=134와 151에서 모두 안정적이면 quick_gelu + tie_embeddings=True를 새 기준선으로 채택하고 silu 또는 gelu_exact를 같은 seed=151 조건에서 단일축 비교한다. / 과적합 시 seed=134에서 quick_gelu가 overfit_risk로 돌아가거나 validation이 악화되면 quick_gelu 효과는 seed 민감성이 있다고 보고 run 008 설정을 추가 seed로 반복하거나 gelu 기준으로 되돌린다.

## run 010 - 2026-06-02T19:43:28+00:00

- 보고서: `docs/train/runs/run_010.md`
- 이번 가설: silu 활성함수 단일축 비교: quick_gelu는 seed=151에서 best를 만들었지만 seed=134에서는 overfit_risk로 돌아갔다. 같은 seed=151, tie_embeddings=True 기준선에서 activation_name만 silu로 바꾸면 부드러운 gate-like 비선형성이 validation 성능과 overfit_score를 quick_gelu 수준으로 유지하거나 더 안정화할 수 있다.
- 근거: run 008은 quick_gelu, seed=151에서 final_val_loss=5.7546, overfit_score=0.1394, fit_status=generalizing으로 현재 best다. 그러나 run 009는 같은 quick_gelu를 seed=134로 바꾸자 fit_status가 overfit_risk로 돌아갔다. seed 134 계열은 gelu와 quick_gelu 모두 overfit_score가 높았으므로, 지금은 같은 seed=151에서 activation family를 비교해 함수 자체의 상대 효과를 분리하는 편이 더 해석 가능하다. silu는 LLM에서 자주 쓰이는 부드러운 activation 계열이고 구조와 parameter_count를 바꾸지 않으므로, quick_gelu와 동일 조건에서 안전하게 비교할 수 있다.
- 바꾼 변수: `{"activation_name": "silu"}`
- 기대 결과: final_val_loss가 5.74~5.82 범위에 머물고 fit_status가 generalizing 또는 mixed 이상이면 silu를 유망 후보로 본다. overfit_score가 run 008의 0.1394보다 낮거나 final_generalization_gap이 0.047 이하이면 quick_gelu보다 더 안정적인 activation 후보가 된다.
- 실제 결과: final_val_loss=5.756640195846558, gap=0.047593116760253906, overfit_score=0.14122319221496582, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0476, overfit_score=0.1412. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 silu가 quick_gelu와 비슷하거나 더 좋으면 같은 silu 설정을 seed=134 또는 새 seed로 반복해 seed 안정성을 확인한다. / 과적합 시 silu가 validation/gap을 악화시키면 quick_gelu를 seed=151 기준 best로 유지하고 gelu_exact 또는 quick_gelu 추가 seed 반복으로 GELU 계열 내부 비교를 이어간다.

## run 011 - 2026-06-02T19:48:53+00:00

- 보고서: `docs/train/runs/run_011.md`
- 이번 가설: gelu_exact 활성함수 단일축 비교: run 008의 quick_gelu가 현재 best이고 run 010의 silu는 generalizing이지만 validation loss와 overfit_score가 약간 나빴다. 같은 seed=151, tie_embeddings=True 기준선에서 activation_name만 gelu_exact로 바꾸면 GELU 계열 내부에서 근사 quick_gelu의 이득이 계산 근사/속도 효과인지, exact GELU에서도 유지되는 안정성인지 분리해 확인할 수 있다.
- 근거: 최근 추세는 seed=151에서 tie_embeddings=True 계열이 generalizing으로 안정되고, activation만 바꾼 run 007 gelu, run 008 quick_gelu, run 010 silu가 모두 비슷한 validation 범위에 머무는 모습을 보인다. run 008은 final_val_loss=5.754559, final_generalization_gap=0.046932, overfit_score=0.139379로 best이고, run 010 silu는 final_val_loss=5.756640, gap=0.047593, overfit_score=0.141223으로 안정적이지만 best는 아니다. 따라서 capacity나 regularization을 새로 섞기보다 동일 seed와 동일 하드웨어 친화 설정에서 gelu_exact만 비교하면 GELU 계열의 의미를 가장 해석 가능하게 좁힐 수 있다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: final_val_loss가 5.74~5.82 범위에 머물고 final_generalization_gap이 0.05 이하이면 GELU 계열은 안정적이라고 본다. overfit_score가 run 008의 0.139379보다 낮거나 final_val_loss가 5.754559보다 낮으면 gelu_exact를 새 후보로 삼고, 비슷하지만 느리면 quick_gelu를 속도와 성능의 균형 후보로 유지한다.
- 실제 결과: final_val_loss=5.754852056503296, gap=0.04696774482727051, overfit_score=0.13946986198425293, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0470, overfit_score=0.1395. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 gelu_exact가 quick_gelu보다 낮은 final_val_loss 또는 overfit_score를 만들면 seed=134 또는 새 seed=202로 반복해 activation 효과가 seed에 강건한지 확인한다. / 과적합 시 gelu_exact가 overfit_risk이거나 run 008보다 validation/gap을 악화시키면 quick_gelu를 GELU 계열 best로 유지하고, 다음에는 quick_gelu seed=202 반복으로 seed variance를 추정한다.

## run 012 - 2026-06-02T19:53:28+00:00

- 보고서: `docs/train/runs/run_012.md`
- 이번 가설: quick_gelu seed variance 검증: run 008은 seed=151에서 현재 best였고 gelu_exact(run 011)는 거의 같은 수준이지만 미세하게 뒤졌다. quick_gelu의 우세가 특정 seed의 우연인지 확인하기 위해, 구조와 학습 조건은 run 008과 동일하게 유지하고 seed만 202로 바꿔 validation loss, gap, overfit_score의 흔들림을 측정한다.
- 근거: 최근 seed=151 activation 비교에서 quick_gelu(run 008), gelu(run 007), gelu_exact(run 011)는 모두 generalizing이며 매우 근접했다. 그중 quick_gelu는 final_val_loss=5.754559, final_generalization_gap=0.046932, overfit_score=0.139379로 best이고 처리량도 균형이 좋다. 반면 seed=134의 quick_gelu(run 009)는 final_val_loss는 비슷했지만 overfit_score=0.173343, fit_status=overfit_risk로 돌아갔다. 따라서 다음 의사결정은 새 activation을 더 늘리기보다, quick_gelu가 새 seed에서도 generalizing 범위와 낮은 overfit_score를 유지하는지 확인하는 것이 더 의미 있다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: final_val_loss가 5.74~5.82 범위에 있고 final_generalization_gap이 0.05 이하, overfit_score가 0.15 이하이면 quick_gelu의 seed 안정성이 강화된다. final_val_loss가 best보다 약간 높더라도 fit_status가 generalizing이면 activation 결론은 유지하고 seed 평균 관점으로 해석한다.
- 실제 결과: final_val_loss=5.769758224487305, gap=0.04903435707092285, overfit_score=0.18662023544311523, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0490, overfit_score=0.1866. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 seed=202에서도 generalizing이면 quick_gelu를 현재 기본 후보로 유지하고, 다음에는 learning_rate 또는 dropout을 작은 폭으로 단일축 조정해 overfit_score를 더 낮출 수 있는지 본다. / 과적합 시 seed=202가 overfit_risk이면 quick_gelu 효과는 seed 민감성이 크다고 보고, 다음에는 같은 조건에서 seed 반복을 한 번 더 하거나 drop_rate=0.15를 결합하지 않고 단일축으로 비교해 regularization 필요성을 분리한다.

## run 013 - 2026-06-02T19:58:38+00:00

- 보고서: `docs/train/runs/run_013.md`
- 이번 가설: drop_rate 정규화 단일축 테스트: quick_gelu seed=202 반복(run 012)은 final_val_loss가 악화되고 overfit_score가 0.1866까지 상승해 seed 민감성과 과적합 위험을 드러냈다. 같은 seed=202와 quick_gelu 설정을 유지하고 drop_rate만 0.10에서 0.15로 올리면 train 쪽 과도한 개선을 누그러뜨려 overfit_score와 train_val_improvement_gap을 낮출 수 있다.
- 근거: run 008 quick_gelu seed=151은 final_val_loss=5.754559, gap=0.046932, overfit_score=0.139379로 현재 best다. 그러나 같은 quick_gelu 계열에서 seed=134(run 009)와 seed=202(run 012)는 overfit_risk로 돌아갔다. 특히 run 012는 final_val_loss=5.769758, gap=0.049034, train_val_improvement_gap=0.068793, overfit_score=0.186620으로 train 개선이 validation보다 크게 앞섰다. 이전 drop_rate=0.20 실험(run 003)은 tie_embeddings=False 기준선에서 validation을 악화시켰으므로, 이번에는 더 작은 0.15를 tie_embeddings=True + quick_gelu + seed=202 조건에 단일축으로 적용해 regularization이 seed 202의 과적합을 완화하는지 분리해서 본다.
- 바꾼 변수: `{"drop_rate": 0.15}`
- 기대 결과: 성공 기준은 final_val_loss가 run 012의 5.769758보다 낮아지거나 최소한 5.74~5.82 범위에 머물면서 overfit_score가 0.15 이하로 내려가는 것이다. final_generalization_gap은 0.05 이하를 유지하고, train_val_improvement_gap이 run 012의 0.068793보다 줄어들면 dropout 0.15가 seed 202의 과적합 완화에 유효하다고 본다.
- 실제 결과: final_val_loss=5.774077653884888, gap=0.04883980751037598, overfit_score=0.1860365867614746, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0488, overfit_score=0.1860. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 drop_rate=0.15가 seed=202에서 generalizing으로 바꾸면 같은 설정을 seed=151 또는 seed=134에서 반복해 dropout 정규화가 평균적으로 유리한지 확인한다. / 과적합 시 drop_rate=0.15에서도 overfit_risk가 유지되면 dropout 단독으로는 부족하다고 보고, 다음에는 learning_rate를 0.0002로 낮추거나 max_steps를 줄이는 최적화 조건 단일축 실험으로 넘어간다.

## run 014 - 2026-06-02T20:03:27+00:00

- 보고서: `docs/train/runs/run_014.md`
- 이번 가설: learning_rate 하향 최적화 단일축 테스트: quick_gelu seed=202 계열은 run 012와 run 013에서 모두 overfit_risk였고, drop_rate=0.15는 validation loss를 더 악화시켰다. dropout을 기준값 0.10으로 되돌리고 learning_rate만 0.0003에서 0.0002로 낮추면 train loss가 너무 빠르게 내려가며 validation 개선을 앞지르는 현상을 줄여 overfit_score와 train_val_improvement_gap을 낮출 수 있다.
- 근거: run 012(seed=202, quick_gelu, drop_rate=0.10)는 final_val_loss=5.769758, gap=0.049034, train_val_improvement_gap=0.068793, overfit_score=0.186620으로 overfit_risk였다. run 013은 같은 seed에서 drop_rate만 0.15로 올렸지만 final_val_loss=5.774078로 더 나빠졌고 overfit_score도 0.186037로 거의 줄지 않았다. 따라서 seed=202의 문제는 단순 dropout 부족보다 학습 속도 또는 step당 업데이트 크기와 관련될 가능성이 있다. 구조, activation, parameter_count를 그대로 유지하고 learning_rate만 낮추면 작은 데이터에서 train 쪽으로 치우친 업데이트를 완만하게 만들어 validation 안정성을 회복할 수 있는지 볼 수 있다.
- 바꾼 변수: `{"learning_rate": 0.0002}`
- 기대 결과: 성공 기준은 run 012 대비 final_val_loss가 5.769758 이하로 내려가거나, 최소한 overfit_score가 0.15 이하로 줄어 fit_status가 generalizing 또는 mixed에 가까워지는 것이다. 학습이 너무 느려져 final_val_loss가 크게 악화되면 learning_rate 하향은 underfit 방향으로 판단한다.
- 실제 결과: final_val_loss=5.956626892089844, gap=0.043051958084106445, overfit_score=0.16867303848266602, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0431, overfit_score=0.1687. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 learning_rate=0.0002가 seed=202의 overfit_score를 의미 있게 낮추면 같은 learning_rate를 seed=151의 best 계열에 적용해 best run 008을 넘어서는지 확인한다. / 과적합 시 learning_rate=0.0002에서도 overfit_risk가 유지되면 업데이트 크기만으로는 부족하다고 보고, 다음에는 max_steps를 30으로 줄이는 조기 중단 단일축 실험으로 train_val_improvement_gap을 직접 낮출 수 있는지 확인한다.

## run 015 - 2026-06-02T20:08:30+00:00

- 보고서: `docs/train/runs/run_015.md`
- 이번 가설: max_steps 조기 중단 단일축 테스트: seed=202 quick_gelu 계열은 learning_rate를 낮추면 validation loss가 크게 악화되고, dropout을 올려도 overfit_score가 거의 줄지 않았다. learning_rate를 0.0003으로 되돌리고 max_steps만 40에서 30으로 줄이면 train 쪽 과도한 개선이 누적되기 전에 멈춰 train_val_improvement_gap과 overfit_score를 낮추면서 run 012보다 validation 손실을 덜 악화시킬 수 있다.
- 근거: run 012는 seed=202, quick_gelu, learning_rate=0.0003, max_steps=40에서 final_val_loss=5.769758, train_val_improvement_gap=0.068793, overfit_score=0.186620으로 overfit_risk였다. run 013은 drop_rate만 0.15로 올렸지만 final_val_loss=5.774078로 더 나빠졌고 overfit_score도 거의 줄지 않았다. run 014는 learning_rate를 0.0002로 낮춰 overfit_score를 0.168673까지 낮췄지만 final_val_loss=5.956627로 너무 나빠져 underfit 성격이 강했다. 따라서 같은 learning_rate=0.0003을 유지하면서 학습 길이만 줄이는 조기 중단이, validation 성능을 과도하게 희생하지 않고 train 과개선을 낮출 수 있는지 확인하는 다음 단일축 실험이다.
- 바꾼 변수: `{"max_steps": 30}`
- 기대 결과: 성공 기준은 final_val_loss가 run 012의 5.769758 이하이거나 비슷한 범위에 머물면서 overfit_score가 run 012의 0.186620보다 의미 있게 낮아지는 것이다. 특히 train_val_improvement_gap이 0.068793보다 줄면 max_steps=30이 seed=202의 과적합 누적을 줄였다고 본다. final_val_loss가 5.9 이상으로 악화되면 너무 짧은 학습으로 판단한다.
- 실제 결과: final_val_loss=5.912327766418457, gap=0.05274999141693115, overfit_score=0.19776713848114014, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0527, overfit_score=0.1978. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 max_steps=30이 validation 손실을 유지하면서 overfit_score를 낮추면 같은 조기 중단 설정을 seed=151의 best quick_gelu 계열에 적용해 run 008보다 더 균형 잡힌 후보가 되는지 확인한다. / 과적합 시 max_steps=30에서도 overfit_risk가 유지되면 seed=202은 초기화/셔플 민감성이 큰 케이스로 보고, 다음에는 seed=151 best 계열에서 더 작은 정규화나 조기 중단을 적용해 best 개선 가능성을 탐색한다.

## run 016 - 2026-06-02T20:13:32+00:00

- 보고서: `docs/train/runs/run_016.md`
- 이번 가설: FFN dropout 위치 교체 단일축 테스트: run 008 quick_gelu seed=151이 현재 best지만 overfit_score는 0.139379로 아직 완전히 낮지는 않다. 같은 best 설정에서 ffn_dropout_position만 after_output에서 after_activation으로 바꾸면 FFN hidden activation에 직접 dropout이 걸려 train 쪽 과개선을 조금 줄이고, 구조를 바꾸지 않으면서 final_val_loss와 overfit_score의 균형을 개선할 수 있다.
- 근거: 최근 seed=202 계열은 seed 자체의 민감성이 커서 dropout, learning_rate, max_steps 조정이 모두 validation 또는 overfit_score를 악화시켰다. 반대로 seed=151 계열은 run 007 gelu, run 008 quick_gelu, run 011 gelu_exact가 모두 generalizing이며 매우 안정적이다. 현재 best인 run 008은 final_val_loss=5.754559, final_generalization_gap=0.046932, overfit_score=0.139379이고, activation 비교에서는 quick_gelu가 gelu_exact와 silu보다 약간 낫다. 따라서 activation을 더 바꾸기보다, 사용자가 추가한 실험 축 중 구조 순서를 크게 바꾸지 않는 dropout 위치만 바꿔 hidden FFN 표현에 대한 regularization 위치 효과를 확인하는 것이 해석 가능하다.
- 바꾼 변수: `{"ffn_dropout_position": "after_activation"}`
- 기대 결과: 성공 기준은 final_val_loss가 run 008의 5.754559와 같거나 더 낮고, overfit_score가 0.139379 이하로 내려가는 것이다. final_val_loss가 약간 높아져도 final_generalization_gap과 train_val_improvement_gap이 안정적으로 낮아지면 위치 교체가 regularization 관점에서 후보가 될 수 있다.
- 실제 결과: final_val_loss=5.754159450531006, gap=0.04617869853973389, overfit_score=0.13711988925933838, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0462, overfit_score=0.1371. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 after_activation이 run 008보다 낮은 overfit_score 또는 비슷한 validation loss를 만들면 같은 설정을 seed=134 또는 새 seed로 반복해 dropout 위치 효과가 seed에 강건한지 확인한다. / 과적합 시 after_activation이 validation을 악화시키거나 overfit_score를 낮추지 못하면 after_output을 유지하고, 다음에는 ffn_dropout_position=none으로 dropout 자체의 ablation을 수행해 현재 dropout이 실제로 유효한지 확인한다.

## run 017 - 2026-06-02T20:18:35+00:00

- 보고서: `docs/train/runs/run_017.md`
- 이번 가설: after_activation dropout 위치 효과 seed=134 재현성 검증: run 016은 seed=151에서 ffn_dropout_position=after_activation이 새 best를 만들었다. 같은 설정을 seed=134로 반복하면, 기존 seed=134 quick_gelu 기준선(run 009)의 overfit_risk가 dropout 위치 변경으로 완화되는지 확인할 수 있다.
- 근거: run 016은 seed=151에서 final_val_loss=5.754159, final_generalization_gap=0.046179, overfit_score=0.137120으로 run 008보다 validation과 overfit_score를 모두 개선했다. 하지만 seed=151 계열은 원래도 generalizing이 강했으므로, 위치 교체가 seed 하나의 우연인지 검증해야 한다. seed=134에는 직접 비교군이 있다. run 009는 quick_gelu + tie_embeddings=True + after_output + seed=134에서 final_val_loss=5.755268로 validation은 나쁘지 않았지만 overfit_score=0.173343, fit_status=overfit_risk였다. 따라서 seed=134에서 after_activation만 적용하면 FFN dropout 위치가 과적합 신호를 줄이는지 명확히 비교할 수 있다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 run 009 대비 final_val_loss가 5.755268 근처를 유지하면서 overfit_score가 0.173343보다 의미 있게 낮아지는 것이다. fit_status가 generalizing으로 바뀌거나 train_val_improvement_gap이 줄면 after_activation 위치 효과가 seed=134에서도 재현된 것으로 본다.
- 실제 결과: final_val_loss=5.754857540130615, gap=0.03545176982879639, overfit_score=0.17111051082611084, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0355, overfit_score=0.1711. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 seed=134에서도 after_activation이 overfit_score를 낮추면 dropout 위치 효과를 유망한 기본값 후보로 보고, 다음에는 seed=202처럼 어려운 seed에서 같은 위치를 적용해 강건성을 더 확인한다. / 과적합 시 seed=134에서도 overfit_risk가 유지되면 after_activation 효과는 seed=151에 국한될 수 있으므로, 다음에는 ffn_dropout_position=none ablation으로 dropout 자체의 기여도를 확인한다.

## run 018 - 2026-06-02T20:23:50+00:00

- 보고서: `docs/train/runs/run_018.md`
- 이번 가설: FFN dropout 자체 기여도 ablation: run 016은 quick_gelu + tie_embeddings=True + seed=151에서 ffn_dropout_position=after_activation이 현재 best를 만들었지만, run 017에서는 seed=134 재현성이 충분하지 않았다. 같은 best seed=151 설정에서 ffn_dropout_position만 none으로 바꾸면 dropout regularization 자체가 validation 성능과 overfit_score에 실제로 기여하는지 분리해서 확인할 수 있다.
- 근거: 최근 흐름에서 seed=151 계열은 run 007, 008, 011, 016 모두 generalizing으로 안정적이고, 그중 run 016은 after_activation이 final_val_loss=5.754159와 overfit_score=0.137120으로 가장 좋았다. 반면 seed=134에서는 after_activation이 run 009 대비 final_val_loss와 overfit_score를 아주 조금만 개선했고 fit_status는 overfit_risk로 남았다. 따라서 새 activation이나 capacity 변경을 섞기 전에, best 계열의 seed=151에서 dropout을 완전히 제거해 after_activation의 이득이 dropout 위치 때문인지, dropout 자체가 꼭 필요한지 확인하는 것이 해석 가능하다.
- 바꾼 변수: `{"ffn_dropout_position": "none"}`
- 기대 결과: 성공 기준은 run 016과 비교해 final_val_loss가 5.754159 이하로 내려가거나 거의 같고, overfit_score가 0.137120 이하를 유지하는 것이다. 만약 final_train_loss만 더 낮아지고 final_val_loss 또는 overfit_score가 악화되면 dropout 제거가 train 암기만 키운 것으로 판단한다. validation이 악화되면 after_activation dropout을 best 계열의 기본 후보로 유지한다.
- 실제 결과: final_val_loss=5.752647876739502, gap=0.04623723030090332, overfit_score=0.13729548454284668, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0462, overfit_score=0.1373. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 ffn_dropout_position=none이 run 016보다 낮은 final_val_loss와 안정적 overfit_score를 만들면 같은 none 설정을 seed=134로 반복해 dropout 제거가 seed 전반에 유효한지 확인한다. / 과적합 시 dropout 제거가 overfit_score를 키우거나 validation을 악화시키면 after_activation을 유지하고, 다음에는 attention_impl=sdpa처럼 성능/속도 위주의 안전한 구현 교체 또는 seed=202의 after_activation 강건성 검증으로 넘어간다.

## run 019 - 2026-06-02T20:28:29+00:00

- 보고서: `docs/train/runs/run_019.md`
- 이번 가설: ffn_dropout_position=none seed 강건성 검증: run 018은 seed=151에서 dropout 제거가 새 best를 만들었다. 같은 quick_gelu + tie_embeddings=True + ffn_dropout_position=none 설정을 seed=134로 반복하면, dropout 제거의 이득이 seed=151에만 국한된 우연인지 아니면 seed=134의 기존 overfit_risk까지 완화하는지 확인할 수 있다.
- 근거: run 018은 final_val_loss=5.752648, overfit_score=0.137295, fit_status=generalizing으로 현재 best가 되었고, run 016(after_activation)보다 validation loss가 낮았다. 그러나 seed=134 계열에서는 run 009(after_output)와 run 017(after_activation)이 모두 final_val_loss는 괜찮았지만 overfit_score가 각각 0.173343, 0.171111로 높아 overfit_risk였다. 따라서 이번에는 best 설정에서 seed만 134로 바꿔, dropout 제거가 seed=134에서도 train_val_improvement_gap과 overfit_score를 낮추는지 직접 비교한다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 run 009와 run 017 대비 final_val_loss를 5.755 이하로 유지하면서 overfit_score를 0.171 이하로 낮추거나 fit_status를 generalizing으로 바꾸는 것이다. final_val_loss가 좋아져도 train_val_improvement_gap과 overfit_score가 유지되면 seed=134의 과적합은 dropout 위치/제거만으로 해결되지 않는다고 판단한다.
- 실제 결과: final_val_loss=5.754166841506958, gap=0.03526628017425537, overfit_score=0.1705540418624878, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0353, overfit_score=0.1706. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 seed=134에서도 ffn_dropout_position=none이 overfit_score를 낮추거나 generalizing을 만들면 none 설정을 새 후보로 유지하고, 다음에는 어려운 seed=202에 적용해 seed 전반의 강건성을 확인한다. / 과적합 시 seed=134에서 여전히 overfit_risk라면 dropout 위치/제거보다 seed variance와 데이터 분할/문맥 조건의 영향이 더 큰 것으로 보고, 다음에는 context_length 또는 stride를 단일축으로 조정하거나 seed=151에서 attention_impl=sdpa 속도 검증으로 안전하게 넘어간다.

## run 020 - 2026-06-02T20:33:39+00:00

- 보고서: `docs/train/runs/run_020.md`
- 이번 가설: 학습 윈도우 stride 단일축 테스트: seed=134 계열은 dropout 위치와 제거를 바꿔도 overfit_score가 0.17대에 머물렀다. best 계열의 quick_gelu + tie_embeddings=True + ffn_dropout_position=none 설정을 유지하고 train stride만 64 상당의 기본값에서 32로 줄이면, 겹치는 학습 윈도우가 늘어나 같은 구조에서 더 많은 문맥 시작점을 보게 되어 validation loss와 train_val_improvement_gap의 균형이 개선될 수 있다.
- 근거: run 009(after_output), run 017(after_activation), run 019(none)은 모두 seed=134에서 final_val_loss는 5.755 안팎으로 괜찮지만 overfit_score가 0.170 이상으로 높았다. 특히 run 019는 dropout 제거가 seed=134에서 약간 개선됐지만 fit_status는 여전히 overfit_risk였다. 이는 FFN dropout 위치보다 seed=134의 데이터 순서/윈도우 구성 영향이 클 수 있음을 시사한다. stride는 모델 구조, parameter_count, activation을 바꾸지 않고 학습 샘플의 시작점만 바꾸는 작은 학습 조건 교체이므로 다음 단일축으로 적절하다.
- 바꾼 변수: `{"stride": 32}`
- 기대 결과: 성공 기준은 run 019 대비 final_val_loss를 5.754 이하로 유지하거나 낮추면서 overfit_score를 0.170 이하로 낮추는 것이다. 특히 train_val_improvement_gap이 0.067644보다 줄면 stride 변경이 seed=134의 train/val 개선 불균형을 완화한 것으로 본다. 반대로 final_train_loss만 더 낮아지고 overfit_score가 커지면 overlap이 데이터 중복만 늘려 과적합을 강화한 것으로 판단한다.
- 실제 결과: final_val_loss=5.755429744720459, gap=0.03773176670074463, overfit_score=0.17198169231414795, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0377, overfit_score=0.1720. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 stride=32가 seed=134의 overfit_score를 낮추면 같은 stride를 best seed=151 설정에 적용해 run 018보다 validation과 overfit_score를 동시에 개선하는지 확인한다. / 과적합 시 stride=32가 overfit_score를 키우거나 validation을 악화시키면 겹치는 윈도우가 중복 학습을 늘린 것으로 보고, 다음에는 context_length=48 또는 32처럼 문맥 길이 자체를 줄이는 단일축 실험으로 capacity와 샘플 수의 균형을 확인한다.

## run 021 - 2026-06-02T20:38:26+00:00

- 보고서: `docs/train/runs/run_021.md`
- 이번 가설: context_length 축소 단일축 테스트: seed=134 계열은 dropout 위치, dropout 제거, stride overlap을 바꿔도 overfit_risk가 유지되었다. 같은 quick_gelu + tie_embeddings=True + ffn_dropout_position=none 설정에서 context_length만 64에서 48로 줄이면, 위치 임베딩 부담과 한 샘플당 문맥 암기 범위가 작아지고 학습/검증 윈도우 수가 늘어 seed=134의 train_val_improvement_gap과 overfit_score가 완화될 수 있다.
- 근거: run 019는 ffn_dropout_position=none과 seed=134에서 final_val_loss=5.754167, overfit_score=0.170554로 run 009/017보다 조금 나았지만 여전히 overfit_risk였다. run 020은 stride=32로 겹치는 윈도우를 늘렸으나 final_val_loss=5.755430, overfit_score=0.171982로 오히려 악화되어, 단순 overlap은 중복 학습을 늘린 것으로 보인다. 따라서 다음에는 중복도를 늘리기보다 context_length 자체를 48로 줄여 더 짧은 문맥과 더 많은 독립 윈도우가 과적합 신호를 낮추는지 본다. 32는 변화가 크므로 먼저 48을 선택해 해석 가능성을 유지한다.
- 바꾼 변수: `{"context_length": 48}`
- 기대 결과: 성공 기준은 run 019 대비 final_val_loss가 5.754167 근처 이하로 유지되고 overfit_score가 0.170554보다 낮아지는 것이다. train_val_improvement_gap이 0.067644보다 줄면 context_length 축소가 seed=134의 train/val 개선 불균형을 완화한 것으로 본다. validation loss가 크게 악화되면 문맥이 너무 짧아져 underfit 또는 정보 부족으로 판단한다.
- 실제 결과: final_val_loss=5.724607149759929, gap=-0.0039515097935991506, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0040, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 context_length=48이 overfit_score를 낮추고 validation을 유지하면 같은 context_length를 best seed=151 설정에 적용해 run 018보다 더 좋은 평균 후보인지 확인한다. / 과적합 시 context_length=48에서도 overfit_risk가 유지되거나 validation이 악화되면 context_length=32로 더 줄이는 대신, seed=151 best 설정에서 attention_impl=sdpa를 비교해 구현 속도/동등성 축을 검증하거나 seed=202에 none 설정을 적용해 seed variance를 정리한다.

## run 022 - 2026-06-02T20:43:20+00:00

- 보고서: `docs/train/runs/run_022.md`
- 이번 가설: context_length=48 seed 재현성 검증: run 021은 seed=134에서 context_length를 64에서 48로 줄이자 final_val_loss와 overfit_score가 크게 개선되어 새 best가 되었다. 같은 quick_gelu + tie_embeddings=True + ffn_dropout_position=none + context_length=48 설정을 seed=151로 반복하면, 이 개선이 seed=134의 우연인지 아니면 현재 데이터 규모에서 더 짧은 문맥이 일반적으로 유리한지 확인할 수 있다.
- 근거: run 018은 seed=151, context_length=64에서 final_val_loss=5.752648, overfit_score=0.137295로 기존 best였고, run 021은 seed=134, context_length=48에서 final_val_loss=5.724607, overfit_score=0.0으로 이를 크게 앞질렀다. 또한 run 020의 stride=32는 중복 윈도우를 늘렸지만 overfit_score를 악화시켰기 때문에, 개선 원인은 overlap이 아니라 문맥 길이 축소와 그에 따른 위치 임베딩/샘플 구성 변화일 가능성이 커졌다. 이제 seed만 151로 바꿔 재현성을 확인해야 context_length=48을 새 기본 후보로 믿을 수 있다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 seed=151에서도 final_val_loss가 run 018의 5.752648보다 낮고, overfit_score가 0.137295보다 낮게 유지되는 것이다. run 021만큼 낮지는 않더라도 fit_status가 generalizing이고 train_val_improvement_gap이 안정적이면 context_length=48 효과가 seed 전반에 재현된 것으로 본다.
- 실제 결과: final_val_loss=5.738766193389893, gap=0.005132794380187988, overfit_score=0.011701782544454531, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0051, overfit_score=0.0117. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=151에서도 context_length=48이 개선되면 seed=202에 같은 설정을 적용해 어려운 seed에서도 과적합이 줄어드는지 확인한다. / 과적합 시 seed=151에서 context_length=48이 validation을 악화시키거나 overfit_risk로 돌아가면, run 021은 seed=134 특이 효과일 수 있으므로 context_length=56 같은 중간값 또는 attention_impl=sdpa 구현 비교로 안전한 축을 탐색한다.

## run 023 - 2026-06-02T20:48:27+00:00

- 보고서: `docs/train/runs/run_023.md`
- 이번 가설: context_length=48 어려운 seed 강건성 검증: seed=202는 context_length=64 계열에서 overfit_risk와 validation 악화를 반복했다. 같은 quick_gelu + tie_embeddings=True + ffn_dropout_position=none + context_length=48 설정을 seed=202로 실행하면, 짧은 문맥이 seed=134/151뿐 아니라 어려운 seed에서도 train_val_improvement_gap과 overfit_score를 안정화하는지 확인할 수 있다.
- 근거: run 021(seed=134, context_length=48)은 final_val_loss=5.724607, overfit_score=0.0으로 새 best가 되었고, run 022(seed=151, context_length=48)도 final_val_loss=5.738766, overfit_score=0.011702로 기존 seed=151 context_length=64 best(run 018)보다 뚜렷하게 좋았다. 반면 seed=202의 context_length=64 계열은 run 012/013/014/015에서 dropout, learning_rate, max_steps 조정에도 overfit_risk 또는 underfit 성격이 남았다. 따라서 seed=202에 context_length=48을 적용하면 이 개선 축이 단순 seed 운이 아니라 현재 데이터 규모에 맞는 문맥 길이 효과인지 가장 강하게 검증할 수 있다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 seed=202에서도 final_val_loss가 run 012의 5.769758보다 낮고, overfit_score가 0.186620보다 크게 낮아지는 것이다. fit_status가 generalizing 또는 low-risk가 되면 context_length=48을 현재 데이터 규모의 핵심 기본 후보로 본다. validation은 좋아지지만 overfit_score가 여전히 높으면 seed=202는 별도 regularization 또는 capacity 축이 필요한 것으로 판단한다.
- 실제 결과: final_val_loss=5.7503306070963545, gap=0.03792969385782907, overfit_score=0.1329832871754979, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0379, overfit_score=0.1330. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=202에서도 context_length=48이 generalizing이면, 다음에는 context_length=48을 고정하고 attention_impl=sdpa 또는 activation_name=gelu_exact/quick_gelu 비교처럼 구현/함수 교체 축으로 넘어간다. / 과적합 시 seed=202에서 context_length=48도 overfit_risk이면, context_length 축은 평균적으로 유효하지만 어려운 seed에는 추가 regularization이 필요하다고 보고 drop_rate=0.15 또는 ffn_mult=3을 context_length=48 위에서 단일축으로 재검토한다.

## run 024 - 2026-06-02T20:53:27+00:00

- 보고서: `docs/train/runs/run_024.md`
- 이번 가설: attention_impl=sdpa 구현 교체 단일축 테스트: context_length=48은 seed=134/151/202에서 모두 기존 64보다 좋은 일반화 신호를 만들었다. 이 설정을 고정하고 attention_impl만 manual에서 sdpa로 바꾸면, Transformer 구조 의미를 유지하면서 MPS에서 처리량을 개선하거나 최소한 같은 validation/overfit 품질을 유지할 수 있는지 확인할 수 있다.
- 근거: run 021(seed=134, context_length=48, manual)은 final_val_loss=5.724607, overfit_score=0.0으로 현재 best이고, run 022/023도 context_length=48에서 generalizing을 재현했다. 이제 데이터 문맥 길이 축은 강한 후보로 정리되었으므로, 다음 실험은 구조를 바꾸지 않는 구현 축을 검증하는 것이 안전하다. scaled_dot_product_attention은 attention 계산 경로만 바꾸며 외부 shape와 Transformer block 순서는 유지된다. run 021의 tokens_per_sec는 14695로 상대적으로 낮았기 때문에, sdpa가 품질을 유지하면서 속도를 개선하는지 확인하면 이후 실험 기본 구현 선택에 도움이 된다.
- 바꾼 변수: `{"attention_impl": "sdpa"}`
- 기대 결과: 성공 기준은 final_val_loss와 overfit_score가 run 021과 같은 low-risk/generalizing 범위에 머물고, tokens_per_sec가 manual 대비 의미 있게 개선되거나 최소한 악화되지 않는 것이다. validation이 크게 흔들리면 sdpa의 수치 경로가 이 작은 실험에서 학습 궤적을 바꾸는 것으로 보고 manual을 유지한다.
- 실제 결과: final_val_loss=5.724607149759929, gap=-0.0039515097935991506, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0040, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 sdpa가 품질을 유지하고 속도를 개선하면 context_length=48 + sdpa를 새 기본 후보로 두고 activation_name=gelu_exact 또는 silu를 다시 비교한다. / 과적합 시 sdpa에서 validation 또는 overfit_score가 악화되면 attention_impl=manual을 유지하고, 다음에는 context_length=48 위에서 activation_name=gelu_exact 단일축 비교로 넘어간다.

## run 025 - 2026-06-02T20:59:44+00:00

- 보고서: `docs/train/runs/run_025.md`
- 이번 가설: context_length=48 + sdpa 기준의 activation 단일축 재비교: quick_gelu가 현재 best 계열이지만, sdpa와 짧은 문맥으로 학습 조건이 안정화된 상태에서도 gelu_exact가 validation loss와 overfit_score에서 계속 뒤처지는지 확인한다.
- 근거: run 021은 context_length=48에서 seed=134의 overfit_score를 0.0으로 낮추며 현재 best를 만들었고, run 022와 run 023도 같은 문맥 길이에서 seed 강건성을 보였다. run 024는 attention_impl만 sdpa로 바꿔도 final_val_loss=5.724607, overfit_score=0.0을 그대로 유지하면서 처리량을 개선했다. 이제 context_length=48 + sdpa를 새 기준 후보로 두고, 이전 64-token 조건에서 근소하게 경쟁했던 gelu_exact를 activation_name 단일축으로 다시 비교하면 activation 효과가 문맥 길이와 attention 구현에 독립적인지 해석할 수 있다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: 성공 기준은 run 024의 final_val_loss=5.724607, overfit_score=0.0과 같거나 더 좋은 validation/generalization 균형을 보이는 것이다. final_val_loss가 같더라도 tokens_per_sec가 크게 느려지면 quick_gelu를 기본 후보로 유지한다. final_val_loss가 개선되고 overfit_score가 낮게 유지되면 gelu_exact를 seed=151 또는 seed=202로 반복 검증한다.
- 실제 결과: final_val_loss=5.724879423777263, gap=-0.0038622220357256154, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0039, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 gelu_exact가 run 024와 동등하거나 더 좋은 validation loss와 low overfit_score를 만들면 seed=151에서 같은 context_length=48 + sdpa + gelu_exact 설정을 반복해 activation 효과가 seed에 강건한지 확인한다. / 과적합 시 gelu_exact에서 gap이나 overfit_score가 커지면 quick_gelu + context_length=48 + sdpa를 유지하고, 다음에는 silu 또는 swiglu/geglu처럼 함수 계열이 다른 activation을 parameter_count와 함께 단일축으로 비교한다.

## run 026 - 2026-06-02T21:03:27+00:00

- 보고서: `docs/train/runs/run_026.md`
- 이번 가설: context_length=48 + sdpa 기준의 activation 계열 확장 테스트: quick_gelu와 gelu_exact가 거의 같은 low-overfit 결과를 냈지만 quick_gelu가 아직 best에 가깝다. 같은 구조와 parameter_count를 유지한 채 activation_name만 silu로 바꾸면, 더 부드러운 비단조 계열이 작은 데이터에서 train/val 균형을 개선하거나 최소한 과적합 없이 경쟁 가능한지 확인할 수 있다.
- 근거: run 021은 context_length=48로 줄이면서 final_val_loss=5.724607, overfit_score=0.0의 best를 만들었고, run 024는 attention_impl=sdpa로 바꿔 같은 validation/generalization 품질을 유지했다. run 025의 gelu_exact는 final_val_loss=5.724879와 overfit_score=0.0으로 안전했지만 best를 넘지는 못했다. 따라서 같은 context_length=48 + sdpa + tie_embeddings=True + ffn_dropout_position=none 설정을 고정하고, parameter_count를 바꾸지 않는 activation 단일축으로 silu를 비교하면 smooth activation 계열이 현재 데이터/문맥 조건에서 유리한지 해석할 수 있다.
- 바꾼 변수: `{"activation_name": "silu"}`
- 기대 결과: 성공 기준은 run 024의 final_val_loss=5.724607 및 overfit_score=0.0과 같거나 더 좋은 validation/generalization 균형이다. final_val_loss가 약간 높더라도 overfit_score가 0에 가깝고 tokens_per_sec가 안정적이면 silu는 seed 반복 후보가 된다. final_val_loss가 악화되거나 train_val_improvement_gap이 커지면 quick_gelu를 activation 기본 후보로 유지한다.
- 실제 결과: final_val_loss=5.726201057434082, gap=-0.003001689910888672, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0030, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 silu가 run 024와 동등하거나 더 좋은 validation loss를 유지하면 seed=151에서 같은 설정을 반복해 smooth activation 효과가 seed에 강건한지 확인한다. / 과적합 시 silu에서 gap이나 overfit_score가 커지면 quick_gelu + context_length=48 + sdpa를 유지하고, 다음에는 parameter_count가 달라지는 swiglu/geglu를 별도 gated FFN 축으로 비교하되 parameter_count와 overfit_score를 함께 해석한다.

## run 027 - 2026-06-02T21:08:34+00:00

- 보고서: `docs/train/runs/run_027.md`
- 이번 가설: context_length=48 + sdpa 기준의 gated FFN 단일축 테스트: quick_gelu, gelu_exact, silu는 모두 low-overfit였지만 best를 갱신하지 못했고, silu는 validation이 약간 밀렸다. activation_name만 swiglu로 바꾸면 gated FFN의 value-gate 상호작용이 작은 문맥 조건에서 표현력을 높여 validation loss를 낮출 수 있는지, 그리고 추가 parameter_count가 과적합으로 이어지는지 확인할 수 있다.
- 근거: run 021과 run 024는 context_length=48에서 final_val_loss=5.724607, overfit_score=0.0으로 현재 기준선을 만들었다. run 025의 gelu_exact는 final_val_loss=5.724879, run 026의 silu는 final_val_loss=5.726201로 모두 과적합 없이 안정적이었지만 quick_gelu 기준을 넘지 못했다. 따라서 단순 smooth activation 계열은 충분히 확인되었고, 다음에는 사용자가 구현한 교체 지점 중 구조 순서를 유지하면서도 FFN 내부 함수만 gated 방식으로 바꾸는 swiglu를 테스트할 만하다. swiglu는 parameter_count 증가 가능성이 있으므로 final_val_loss 개선이 overfit_score와 parameter_count 증가를 정당화하는지 함께 판단한다.
- 바꾼 변수: `{"activation_name": "swiglu"}`
- 기대 결과: 성공 기준은 run 024의 final_val_loss=5.724607보다 낮거나 거의 같으면서 overfit_score가 0.05 이하로 유지되는 것이다. parameter_count가 증가할 것이므로, validation loss가 명확히 개선되지 않거나 gap이 커지면 swiglu의 추가 표현력은 이 데이터 규모에서 정당화되지 않는다고 판단한다. tokens_per_sec가 크게 떨어지는지도 함께 본다.
- 실제 결과: final_val_loss=5.748608907063802, gap=-0.015673557917277314, overfit_score=0.012871424357095762, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0157, overfit_score=0.0129. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 swiglu가 validation loss를 낮추고 overfit_score를 낮게 유지하면 seed=151에서 같은 swiglu 설정을 반복해 gated FFN 효과가 seed에 강건한지 확인한다. / 과적합 시 swiglu에서 gap이나 overfit_score가 커지면 gated FFN의 추가 capacity가 과적합을 유도한 것으로 보고 quick_gelu + context_length=48 + sdpa를 유지한다. 다음에는 geglu를 바로 확대하기보다 ffn_mult=3 또는 tie_embeddings/weight_decay 조합으로 gated FFN capacity를 낮춘 재검증을 고려한다.

## run 028 - 2026-06-02T21:13:44+00:00

- 보고서: `docs/train/runs/run_028.md`
- 이번 가설: best 계열로 복귀한 context_length 로컬 서치: run 021/024의 핵심 개선은 activation보다 context_length=48에서 발생했고, 이후 gelu_exact/silu/swiglu는 best를 갱신하지 못했다. quick_gelu + sdpa + tie_embeddings=True + ffn_dropout_position=none을 유지하고 context_length만 48에서 40으로 더 줄이면, 작은 데이터에서 위치 임베딩 부담과 문맥 암기 범위를 더 낮춰 validation loss와 overfit_score가 추가 개선될 수 있는지 확인한다.
- 근거: run 021은 context_length를 64에서 48로 줄이자 final_val_loss=5.724607, overfit_score=0.0의 best를 만들었고, run 024는 sdpa에서도 같은 품질을 유지했다. 반면 run 025(gelu_exact), run 026(silu), run 027(swiglu)은 모두 low-risk였지만 validation loss가 각각 5.724879, 5.726201, 5.748609로 best를 넘지 못했다. 특히 swiglu는 parameter_count를 611072까지 늘렸지만 loss가 악화되어, 지금은 capacity/activation 확장보다 데이터 윈도우 조건을 더 정밀하게 확인하는 것이 타당하다. context_length=40은 48보다 변화가 작고, 구조 순서를 바꾸지 않으며, MPS에서 더 가볍게 검증할 수 있는 로컬 탐색점이다.
- 바꾼 변수: `{"activation_name": "quick_gelu", "attention_impl": "sdpa", "context_length": 40}`
- 기대 결과: 성공 기준은 run 024의 final_val_loss=5.724607보다 낮거나 거의 같고, overfit_score가 0.05 이하로 유지되는 것이다. context_length=40에서 validation loss가 개선되면 더 짧은 문맥이 현재 corpus에 더 맞는다는 신호다. 반대로 final_val_loss가 48-token 기준보다 악화되면 문맥 정보 손실 또는 샘플 구성 변화가 손해라고 보고 context_length=48을 유지한다.
- 실제 결과: final_val_loss=5.794515291849772, gap=0.04809276262919138, overfit_score=0.17611058553060044, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0481, overfit_score=0.1761. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 context_length=40이 validation loss를 개선하거나 동등하게 유지하면 seed=151에서 같은 quick_gelu + sdpa + context_length=40 설정을 반복해 짧은 문맥 효과가 seed에 강건한지 확인한다. / 과적합 시 context_length=40에서 gap이나 overfit_score가 커지거나 validation이 악화되면 context_length=48을 기본 후보로 유지하고, 다음에는 context_length=56처럼 48과 64 사이의 중간값을 확인해 최적 문맥 길이 범위를 좁힌다.

## run 029 - 2026-06-02T21:18:28+00:00

- 보고서: `docs/train/runs/run_029.md`
- 이번 가설: context_length 경계 확인 실험: context_length=40은 overfit_risk와 validation 악화를 만들었고, context_length=48은 현재 best이다. quick_gelu + sdpa + tie_embeddings=True + ffn_dropout_position=none을 유지한 채 context_length만 56으로 늘리면, 48보다 조금 더 긴 문맥이 정보량을 보강하면서도 64-token 계열의 과적합 위험으로 돌아가지 않는지 확인할 수 있다.
- 근거: run 021/024는 context_length=48에서 final_val_loss=5.724607, overfit_score=0.0으로 가장 좋은 균형을 만들었다. run 028은 같은 best 계열에서 context_length를 40으로 줄였지만 final_val_loss=5.794515, gap=0.048093, overfit_score=0.176111로 high-risk가 되어 너무 짧은 문맥은 validation 구성과 train/val 균형을 해친다는 증거가 되었다. 반대로 64-token seed=134 계열은 이전에 overfit_risk가 반복되었다. 따라서 56은 48과 64 사이의 경계점으로, 구조와 함수 선택을 유지하면서 문맥 길이의 안정 범위를 좁히는 해석 가능한 단일축 실험이다.
- 바꾼 변수: `{"context_length": 56}`
- 기대 결과: 성공 기준은 final_val_loss가 run 024의 5.724607에 근접하거나 더 낮고, overfit_score가 0.05 이하로 유지되는 것이다. 56에서 validation이 48보다 나빠지지만 overfit_score가 낮으면 48이 더 좋은 local optimum으로 판단한다. gap과 overfit_score가 커지면 긴 문맥이 다시 64-token 계열의 overfit_risk로 접근한다고 본다.
- 실제 결과: final_val_loss=5.736375331878662, gap=0.0521618127822876, overfit_score=0.15642964839935303, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0522, overfit_score=0.1564. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 context_length=56이 48과 동등하거나 더 좋은 validation/generalization 균형을 만들면 seed=151로 반복해 긴 문맥의 안정성이 seed에 강건한지 확인한다. / 과적합 시 context_length=56에서 overfit_score가 커지거나 validation이 악화되면 context_length=48을 기본 후보로 고정하고, 다음에는 seed 반복 또는 learning_rate/weight_decay 같은 optimization 축을 48 기준 위에서 탐색한다.

## run 030 - 2026-06-02T21:23:32+00:00

- 보고서: `docs/train/runs/run_030.md`
- 이번 가설: context_length=48 기준의 학습 길이 단일축 테스트: context_length 40과 56은 모두 overfit_risk였고, activation/gated FFN 교체도 best를 넘지 못했다. 따라서 quick_gelu + sdpa + context_length=48을 고정하고 max_steps만 40에서 60으로 늘리면, 현재 낮은 overfit_score를 유지하면서 train/val loss를 더 낮출 수 있는지 확인한다.
- 근거: run 021/024는 context_length=48에서 final_val_loss=5.724607, overfit_score=0.0으로 현재 best를 만들었다. 이후 run 025 gelu_exact, run 026 silu, run 027 swiglu는 activation/capacity 축에서 best를 갱신하지 못했고, run 028(context_length=40)과 run 029(context_length=56)는 high-risk로 무너졌다. 즉 현재 탐색의 가장 신뢰할 수 있는 기준은 quick_gelu + context_length=48 + sdpa이다. 이 기준은 final_generalization_gap이 음수에 가깝고 overfit_score가 0이어서, 40 step에서 아직 조금 덜 학습되었을 가능성이 있다. max_steps=60은 MPS에서 여전히 짧은 회차이며, 과적합 감시 하에 학습 길이만 늘리는 해석 가능한 optimization 축이다.
- 바꾼 변수: `{"max_steps": 60}`
- 기대 결과: 성공 기준은 final_val_loss가 run 024의 5.724607보다 낮아지고, overfit_score가 0.05 이하로 유지되는 것이다. train loss만 낮아지고 validation gap 또는 train_val_improvement_gap이 커지면 60 step은 과학적으로는 학습 지속 효과를 확인했지만 실험 기본값으로는 위험하다고 본다.
- 실제 결과: final_val_loss=5.588832537333171, gap=0.021294991175333955, overfit_score=0.0696878433227548, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0213, overfit_score=0.0697. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 max_steps=60이 validation loss를 낮추면서 low-risk를 유지하면 seed=151에서 같은 60-step 설정을 반복해 학습 길이 이득이 seed에 강건한지 확인한다. / 과적합 시 max_steps=60에서 overfit_score가 커지거나 validation이 악화되면 max_steps=40을 기본값으로 되돌리고, 다음에는 learning_rate=0.00025 또는 weight_decay=0.02처럼 더 완만한 optimization/regularization 축을 context_length=48 위에서 단일축으로 테스트한다.

## run 031 - 2026-06-02T21:28:28+00:00

- 보고서: `docs/train/runs/run_031.md`
- 이번 가설: max_steps=60 seed 재현성 검증: run 030은 context_length=48 + quick_gelu + sdpa 기준에서 max_steps를 60으로 늘리자 validation loss가 크게 개선되고 generalizing을 유지했다. 같은 설정을 seed=151로 반복하면, 이 개선이 seed=134의 우연인지 아니면 학습 길이 증가가 안정적으로 유효한지 확인할 수 있다.
- 근거: run 030은 final_val_loss=5.588833, final_generalization_gap=0.021295, overfit_score=0.069688, fit_status=generalizing으로 새 best를 만들었다. 기존 40-step 기준의 run 022(seed=151, context_length=48)는 final_val_loss=5.738766, overfit_score=0.011702로 안정적이었지만 충분히 학습되지 않았을 가능성이 있다. context_length 40/56 및 activation/gated FFN 축이 모두 best를 넘지 못했으므로, 현재 가장 유망한 축은 context_length=48을 고정한 학습 길이 증가다. seed=151 반복은 max_steps=60 이득의 재현성과 과적합 위험을 동시에 확인하는 가장 직접적인 다음 실험이다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 seed=151에서도 final_val_loss가 기존 seed=151 40-step run 022의 5.738766보다 명확히 낮아지고, overfit_score가 0.10 이하 또는 fit_status=generalizing을 유지하는 것이다. validation은 개선되지만 overfit_score가 크게 오르면 60 step은 추가 regularization과 함께 써야 한다고 판단한다.
- 실제 결과: final_val_loss=5.595870176951091, gap=0.035879413286845185, overfit_score=0.10394163926442612, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0359, overfit_score=0.1039. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=151에서도 max_steps=60이 validation loss를 낮추고 generalizing을 유지하면 seed=202로 반복해 어려운 seed에서도 학습 길이 효과가 유지되는지 확인한다. / 과적합 시 seed=151에서 max_steps=60이 overfit_risk로 바뀌면 max_steps=60은 seed 민감성이 있다고 보고, context_length=48 기준에서 learning_rate=0.00025 또는 weight_decay=0.02를 단일축으로 추가해 과적합을 완화한다.

## run 032 - 2026-06-02T21:33:28+00:00

- 보고서: `docs/train/runs/run_032.md`
- 이번 가설: max_steps=60 어려운 seed 강건성 검증: seed=134와 seed=151에서 context_length=48 + quick_gelu + sdpa + max_steps=60은 validation loss를 크게 낮추며 generalizing을 유지했다. 같은 설정을 seed=202로 반복하면, 이전에 더 불안정했던 seed에서도 학습 길이 증가가 validation 성능을 개선하면서 과적합을 감당할 수 있는지 확인한다.
- 근거: run 030(seed=134)은 final_val_loss=5.588833, overfit_score=0.069688로 새 best가 되었고, run 031(seed=151)은 final_val_loss=5.595870, overfit_score=0.103942로 best에는 못 미쳤지만 기존 seed=151 40-step run 022의 5.738766보다 크게 개선되었다. seed=202는 40-step context_length=48 run 023에서 final_val_loss=5.750331, overfit_score=0.132983으로 상대적으로 어려운 seed였다. 따라서 seed=202에서 60 step을 적용하면, 학습 길이 증가가 seed 전반의 under-training을 해결하는지 아니면 어려운 seed에서 overfit_risk를 키우는지 판단할 수 있다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 seed=202에서도 final_val_loss가 기존 run 023의 5.750331보다 명확히 낮아지고, fit_status가 generalizing을 유지하는 것이다. overfit_score가 0.15 이하이면 어려운 seed에서도 감당 가능한 개선으로 본다. validation은 좋아지지만 overfit_score가 0.15를 넘으면 60 step은 유효하지만 추가 regularization 후보가 필요하다.
- 실제 결과: final_val_loss=5.601549784342448, gap=0.004055897394816377, overfit_score=0.031361897786459814, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0041, overfit_score=0.0314. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=202에서도 max_steps=60이 generalizing을 유지하며 validation loss를 낮추면 max_steps=60을 새 기본 학습 길이 후보로 두고, 다음에는 max_steps=80 또는 learning_rate 조정으로 더 긴 학습의 한계를 확인한다. / 과적합 시 seed=202에서 max_steps=60이 overfit_risk를 만들면 max_steps=60은 효과가 있지만 seed에 따라 과적합이 커지는 것으로 보고, context_length=48 + max_steps=60 위에서 weight_decay=0.02 또는 drop_rate=0.12를 단일축으로 테스트한다.

## run 033 - 2026-06-02T21:38:23+00:00

- 보고서: `docs/train/runs/run_033.md`
- 이번 가설: max_steps=80 학습 한계 테스트: context_length=48 + quick_gelu + sdpa 기준에서 max_steps=60은 seed=134/151/202 모두 validation을 크게 개선하며 generalizing을 유지했다. seed=202에서 max_steps만 80으로 늘리면, 더 긴 학습이 validation loss를 추가로 낮출 수 있는지 또는 과적합 신호가 급격히 커지는지 확인할 수 있다.
- 근거: run 030(seed=134), run 031(seed=151), run 032(seed=202)은 모두 context_length=48 + quick_gelu + sdpa + max_steps=60에서 기존 40-step 기준보다 validation loss를 크게 낮췄고 fit_status=generalizing을 유지했다. 특히 run 032는 어려운 seed=202에서도 final_val_loss=5.601550, gap=0.004056, overfit_score=0.031362로 매우 안정적이었다. 따라서 다음 자연스러운 질문은 60 step이 아직 under-training인지, 아니면 더 길게 학습하면 gap이 커지는지다. seed=202는 overfit_score가 가장 낮게 나온 최신 후보라 max_steps=80의 리스크를 관찰하기 좋은 시작점이다.
- 바꾼 변수: `{"max_steps": 80}`
- 기대 결과: 성공 기준은 final_val_loss가 run 032의 5.601550보다 낮고, overfit_score가 0.10 이하 또는 fit_status=generalizing을 유지하는 것이다. validation loss가 조금 낮아져도 gap과 overfit_score가 크게 커지면 80 step은 과학적으로는 학습 한계를 보여주지만 기본 후보로는 보류한다.
- 실제 결과: final_val_loss=5.553315162658691, gap=0.010401010513305664, overfit_score=0.050397237141927675, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0104, overfit_score=0.0504. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 max_steps=80이 seed=202에서 validation을 개선하고 generalizing을 유지하면 seed=134 또는 seed=151로 반복해 80-step 이득이 seed에 강건한지 확인한다. / 과적합 시 max_steps=80에서 overfit_score가 커지거나 validation이 악화되면 max_steps=60을 기본 후보로 유지하고, 다음에는 learning_rate=0.00025 또는 weight_decay=0.02를 max_steps=80과 결합하기 전에 단일축으로 검증한다.

## run 034 - 2026-06-02T21:44:01+00:00

- 보고서: `docs/train/runs/run_034.md`
- 이번 가설: max_steps=80 seed=134 재현성 검증: run 033에서 seed=202는 max_steps=80으로 validation loss를 5.553315까지 낮추면서 gap=0.010401, overfit_score=0.050397의 generalizing 상태를 유지했다. 같은 context_length=48 + quick_gelu + sdpa 기준에서 seed만 134로 바꾸면, 80-step 이득이 특정 seed의 우연인지 아니면 현재 best 계열의 안정적인 학습 길이 효과인지 확인할 수 있다.
- 근거: 최근 추세는 context_length=48이 40/56보다 명확히 안정적이고, quick_gelu가 gelu_exact/silu/swiglu보다 validation과 처리량 균형이 좋으며, sdpa가 품질을 유지하면서 MPS 처리량을 개선한다는 쪽으로 모였다. run 030/031/032에서 max_steps=60은 seed=134/151/202 모두 validation을 크게 낮췄고, run 033은 seed=202에서 max_steps=80도 추가 개선을 만들었다. 다만 80-step은 아직 seed=202에서만 확인되었으므로, seed=134 반복이 가장 직접적인 강건성 검증이다. seed=134는 60-step run 030에서 pure val이 강했던 seed라, 80-step이 더 좋아지는지 혹은 gap이 커지는지 관찰 가치가 크다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed=134에서도 final_val_loss가 run 030의 60-step 결과 5.588833보다 낮아지고, overfit_score가 0.12 이하 또는 fit_status=generalizing을 유지하는 것이다. final_val_loss가 좋아져도 final_generalization_gap과 train_val_improvement_gap이 크게 커지면 80-step은 추가 regularization이 필요한 학습 길이로 본다.
- 실제 결과: final_val_loss=5.554664134979248, gap=0.04753577709197998, overfit_score=0.14841020107269287, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0475, overfit_score=0.1484. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 seed=134에서도 80-step이 validation을 개선하고 generalizing을 유지하면 seed=151로 같은 max_steps=80을 반복해 세 seed 모두에서 강건한지 확인한다. 세 seed가 안정적이면 다음에는 max_steps=100의 한계 테스트보다 먼저 learning_rate=0.00025 또는 weight_decay=0.02를 80-step 기준에서 비교해 더 낮은 validation과 낮은 overfit_score의 균형을 찾는다. / 과적합 시 seed=134에서 80-step이 overfit_risk를 만들거나 gap이 크게 증가하면 max_steps=60을 더 안전한 기본 후보로 유지한다. 이후에는 max_steps=80을 계속 밀기보다 weight_decay=0.02, drop_rate=0.12, learning_rate=0.00025 중 하나를 단일축으로 붙여 긴 학습의 과적합 완화 가능성을 검증한다.

## run 035 - 2026-06-02T21:48:30+00:00

- 보고서: `docs/train/runs/run_035.md`
- 이번 가설: max_steps=80 seed=134 과적합 완화 단일축 테스트: run 034는 seed=134에서 validation loss를 5.554664까지 낮췄지만 final_generalization_gap=0.047536, overfit_score=0.148410으로 overfit_risk가 되었다. 같은 80-step 설정에서 weight_decay만 0.01에서 0.02로 올리면, train loss 과도 개선을 억제해 validation 성능을 유지하면서 gap과 overfit_score를 낮출 수 있는지 확인한다.
- 근거: run 033(seed=202, max_steps=80)은 generalizing으로 best가 되었지만 run 034(seed=134, max_steps=80)는 거의 같은 final_val_loss에도 gap과 overfit_score가 크게 증가했다. 이는 max_steps=80 자체가 무조건 나쁜 것이 아니라 seed=134에서 긴 학습이 train 쪽으로 더 빠르게 맞춰지는 문제일 가능성을 보여준다. context_length=48, quick_gelu, sdpa, tie_embeddings=True, ffn_dropout_position=none은 최근 실험에서 가장 안정적인 구조/함수 조합이므로 유지하고, 모델 구조나 activation을 바꾸기 전에 regularization 단일축인 weight_decay를 먼저 확인하는 것이 해석 가능하다.
- 바꾼 변수: `{"weight_decay": 0.02}`
- 기대 결과: 성공 기준은 run 034 대비 final_generalization_gap과 overfit_score가 낮아지고, final_val_loss가 5.57 이하로 유지되는 것이다. 특히 overfit_score가 0.12 이하로 내려가거나 fit_status가 generalizing으로 돌아오면 80-step에는 약한 weight_decay 강화가 필요하다는 가설을 지지한다. validation loss가 악화되고 gap만 줄면 under-training 또는 과도한 regularization으로 해석한다.
- 실제 결과: final_val_loss=5.554707686106364, gap=0.04750410715738962, overfit_score=0.1483151912689218, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0475, overfit_score=0.1483. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 weight_decay=0.02가 seed=134의 overfit_score를 낮추면서 validation을 유지하면, 같은 설정을 seed=202 또는 seed=151에 반복해 regularization 이득이 seed 전반에 해롭지 않은지 확인한다. 이후 best 후보는 max_steps=80 + weight_decay=0.02 계열과 run 033의 weight_decay=0.01 계열을 score 기준으로 비교한다. / 과적합 시 weight_decay=0.02에서도 overfit_risk가 유지되면 weight_decay만으로는 긴 학습의 seed=134 과적합을 막기 어렵다고 보고, 다음에는 learning_rate=0.00025 또는 drop_rate=0.12를 단일축으로 테스트한다. validation이 크게 나빠지면 max_steps=60을 더 안전한 기본 학습 길이로 되돌린다.

## run 036 - 2026-06-02T21:53:32+00:00

- 보고서: `docs/train/runs/run_036.md`
- 이번 가설: max_steps=80 seed=134 learning_rate 완화 단일축 테스트: run 034와 run 035는 validation loss가 5.55대까지 내려갔지만 train loss가 빠르게 낮아지며 gap≈0.0475, overfit_score≈0.148의 overfit_risk를 반복했다. weight_decay=0.02는 거의 효과가 없었으므로, 이번에는 weight_decay를 0.01로 되돌리고 learning_rate만 0.0003에서 0.00025로 낮춰 긴 학습의 train 편향을 완화할 수 있는지 확인한다.
- 근거: 최근 evidence는 context_length=48 + quick_gelu + sdpa + tie_embeddings=True + ffn_dropout_position=none 조합이 가장 안정적인 구조/함수 후보임을 보여준다. max_steps=60은 seed=134/151/202 모두 개선됐고, max_steps=80은 seed=202에서 best generalizing을 만들었지만 seed=134에서는 overfit_risk가 되었다. run 035에서 weight_decay만 올려도 gap과 overfit_score가 거의 줄지 않았으므로, 단순 가중치 감쇠보다 optimization 속도가 seed=134 과적합의 원인일 가능성이 크다. learning_rate를 낮추는 것은 구조를 유지하면서 학습 궤적만 바꾸는 해석 가능한 단일축이다.
- 바꾼 변수: `{"learning_rate": 0.00025}`
- 기대 결과: 성공 기준은 run 034 대비 final_generalization_gap과 overfit_score가 의미 있게 낮아지고, final_val_loss가 5.57 이하로 유지되는 것이다. overfit_score가 0.12 이하로 내려가거나 fit_status가 generalizing으로 돌아오면 80-step seed=134 문제는 learning_rate 속도 문제였다는 가설을 지지한다. final_val_loss가 5.59 이상으로 악화되면 learning_rate를 낮춘 만큼 under-training이 생긴 것으로 보고 max_steps=60이나 다른 regularization 축을 우선한다.
- 실제 결과: final_val_loss=5.575037797292073, gap=0.03290029366811087, overfit_score=0.10450375080108554, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0329, overfit_score=0.1045. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 learning_rate=0.00025가 seed=134에서 gap을 줄이고 validation을 유지하면 같은 설정을 seed=202 또는 seed=151에 반복해 best run033의 성능을 더 낮은 overfit_score로 재현할 수 있는지 확인한다. 이후에는 0.00025와 0.0003 사이의 중간값을 탐색하기보다 seed 반복을 먼저 해서 안정성을 본다. / 과적합 시 learning_rate=0.00025에서도 overfit_risk가 유지되면 80-step seed=134의 문제는 학습 속도만으로 해결되지 않는 것으로 보고, max_steps=60을 안전 기본 후보로 되돌리거나 drop_rate=0.12를 단일축으로 테스트한다. validation이 크게 악화되면 낮은 learning_rate는 under-training으로 판단한다.

## run 037 - 2026-06-02T21:58:48+00:00

- 보고서: `docs/train/runs/run_037.md`
- 이번 가설: max_steps=80 seed=134 learning_rate 중간값 균형 테스트: run 034는 learning_rate=0.0003에서 final_val_loss=5.554664로 강했지만 overfit_risk였고, run 036은 learning_rate=0.00025에서 generalizing으로 회복했지만 final_val_loss가 5.575038로 악화되었다. 같은 설정에서 learning_rate만 0.000275로 두면, 0.0003의 validation 성능을 일부 유지하면서 0.00025가 보여준 gap/overfit_score 완화 효과를 얻을 수 있는지 확인한다.
- 근거: run 035의 weight_decay=0.02는 run 034 대비 gap과 overfit_score를 거의 줄이지 못했으므로 seed=134의 80-step 과적합은 단순 가중치 감쇠보다 optimization 속도와 더 관련 있어 보인다. run 036은 learning_rate를 0.00025로 낮추자 fit_status가 overfit_risk에서 generalizing으로 돌아왔지만 validation loss 손실이 컸다. 따라서 seed를 바꾸기 전에 같은 seed=134에서 0.0003과 0.00025 사이의 중간 learning_rate를 테스트하면, 과적합 완화와 validation 성능 사이의 균형점이 존재하는지 더 직접적으로 알 수 있다. 구조, activation, attention 구현은 모두 유지한다.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: 성공 기준은 final_val_loss가 run 036의 5.575038보다 낮고, run 034/035보다 final_generalization_gap과 overfit_score가 낮으며, fit_status가 generalizing 또는 최소한 overfit_score 0.12 이하를 유지하는 것이다. final_val_loss가 5.56 안팎이고 overfit_score가 0.12 이하이면 0.000275가 seed=134의 균형점 후보가 된다. val은 좋아지지만 overfit_risk가 반복되면 0.0003 쪽 학습 속도 문제가 여전히 남은 것으로 본다.
- 실제 결과: final_val_loss=5.5632913907368975, gap=0.039051850636799834, overfit_score=0.12295842170715243, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0391, overfit_score=0.1230. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 learning_rate=0.000275가 seed=134에서 validation과 overfit_score 균형을 만들면, 같은 설정을 seed=202에 반복해 run 033의 best 성능을 더 안정적인 gap으로 재현할 수 있는지 확인한다. 그 뒤 seed=151 반복으로 평균 후보를 판단한다. / 과적합 시 0.000275에서도 overfit_risk가 유지되면 learning_rate를 낮추는 것만으로는 seed=134의 80-step 균형이 어렵다고 보고, max_steps=60을 더 안전한 기본값으로 유지하거나 drop_rate=0.12 단일축을 테스트한다. validation이 run036보다 악화되면 80-step의 낮은 learning_rate는 under-training으로 판단한다.

## run 038 - 2026-06-02T22:03:35+00:00

- 보고서: `docs/train/runs/run_038.md`
- 이번 가설: learning_rate=0.000275 seed=202 강건성 검증: seed=134에서는 learning_rate=0.000275가 0.0003의 과적합 위험과 0.00025의 validation 손실 사이에서 중간 균형을 만들었다. 같은 max_steps=80 + context_length=48 + quick_gelu + sdpa 기준을 seed=202에 적용하면, best run033의 낮은 validation loss를 유지하면서 generalization gap과 overfit_score를 더 낮추거나 안정화할 수 있는지 확인한다.
- 근거: run 033은 seed=202, learning_rate=0.0003, max_steps=80에서 final_val_loss=5.553315, gap=0.010401, overfit_score=0.050397로 현재 best다. 반면 seed=134에서는 같은 learning_rate=0.0003이 run 034에서 overfit_risk를 만들었고, learning_rate=0.000275는 run 037에서 final_val_loss=5.563291, gap=0.039052, overfit_score=0.122958의 generalizing 중간점을 만들었다. seed=202는 이미 안정적이므로, 낮춘 learning_rate가 불필요하게 underfit을 만들지 아니면 best 후보의 gap을 더 낮추는지 확인하는 것이 다음으로 해석 가능하다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 final_val_loss가 run 033의 5.553315에 근접하거나 더 낮고, overfit_score가 0.05 안팎 이하로 유지되는 것이다. final_val_loss가 5.57 이상으로 악화되면 seed=202에서는 0.0003이 더 적절하고 0.000275는 under-training 쪽이라고 본다. gap과 overfit_score가 좋아져도 validation이 명확히 나빠지면 best 후보는 유지하지 않는다.
- 실제 결과: final_val_loss=5.563548405965169, gap=0.0033559401830034474, overfit_score=0.029262026151021026, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0034, overfit_score=0.0293. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=202에서도 learning_rate=0.000275가 run033과 비슷한 validation에 더 낮은 overfit_score를 만들면 seed=151로 반복해 세 seed 평균 후보로 평가한다. 성공 시 0.000275를 max_steps=80 기본 learning_rate 후보로 올린다. / 과적합 시 seed=202에서 overfit_score가 커지거나 validation이 나빠지면 learning_rate=0.000275는 seed=134용 보정에 가깝다고 보고, best run033 설정을 유지한다. 다음에는 seed=151에서 max_steps=80 learning_rate=0.0003을 먼저 확인하거나 drop_rate=0.12를 seed=134에 단일축으로 테스트한다.

## run 039 - 2026-06-02T22:08:33+00:00

- 보고서: `docs/train/runs/run_039.md`
- 이번 가설: learning_rate=0.000275 seed=151 평균 후보 검증: seed=134와 seed=202에서 learning_rate=0.000275는 max_steps=80 조건의 과적합 위험을 줄이면서 generalizing을 유지했다. 같은 설정을 seed=151에 반복하면, 0.000275가 특정 seed 보정이 아니라 세 seed 평균에서 안정적인 learning_rate 후보인지 판단할 수 있다.
- 근거: run 037(seed=134, lr=0.000275)은 final_val_loss=5.563291, overfit_score=0.122958로 run 034의 overfit_risk를 피하면서 validation 손실도 0.00025보다 줄였다. run 038(seed=202, lr=0.000275)은 final_val_loss=5.563548로 run 033보다 조금 나빴지만 gap=0.003356, overfit_score=0.029262로 매우 안정적이었다. seed=151은 max_steps=60에서 medium-risk generalizing을 보였으므로, max_steps=80과 lr=0.000275의 조합이 평균 성능과 과적합 균형을 유지하는지 확인하기 좋은 남은 seed다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 final_val_loss가 seed=151의 max_steps=60 결과인 run 031의 5.595870보다 낮고, overfit_score가 0.12 이하 또는 fit_status=generalizing을 유지하는 것이다. final_val_loss가 5.56~5.58 범위이고 overfit_score가 0.12 안팎 이하이면 0.000275를 세 seed 평균 후보로 볼 수 있다. validation이 좋아져도 overfit_risk가 되면 seed=151에는 추가 regularization이나 max_steps=60 유지가 필요하다.
- 실제 결과: final_val_loss=5.561801274617513, gap=0.024208386739094756, overfit_score=0.06892855962117483, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0242, overfit_score=0.0689. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=151에서도 learning_rate=0.000275가 generalizing을 유지하면, 세 seed의 평균 validation/overfit_score를 기준으로 lr=0.000275와 run033의 lr=0.0003 best 후보를 비교한다. 이후에는 drop_rate=0.12를 seed=134 또는 seed=151에서 단일축으로 테스트해 0.0003의 낮은 loss와 0.000275의 안정성을 결합할 수 있는지 확인한다. / 과적합 시 seed=151에서 overfit_risk가 생기면 lr=0.000275는 seed=202에는 안정적이지만 평균 후보로는 불충분하다고 보고, max_steps=60을 안전 기준으로 유지하거나 drop_rate=0.12/learning_rate=0.00025 중 하나를 seed=151에서 단일축으로 검토한다.

## run 040 - 2026-06-02T22:16:56+00:00

- 보고서: `docs/train/runs/run_040.md`
- 이번 가설: drop_rate=0.12 seed=134 과적합 완화 테스트: run034는 learning_rate=0.0003, max_steps=80에서 validation loss는 5.554664로 강했지만 gap=0.047536, overfit_score=0.148410의 overfit_risk가 되었다. weight_decay=0.02는 거의 효과가 없었고, learning_rate=0.000275는 안정적이지만 validation loss가 약간 밀렸다. 따라서 learning_rate=0.0003의 낮은 validation 후보를 유지한 채 drop_rate만 0.10에서 0.12로 아주 작게 올리면 train 쪽 과도 적합을 줄이면서 low-loss 영역을 보존할 수 있는지 확인한다.
- 근거: 최근 run037-run039는 learning_rate=0.000275가 세 seed에서 generalizing을 유지한다는 것을 보여줬지만, best run033 및 seed=134의 run034보다 validation loss는 약간 높았다. run034와 run035의 비교는 weight_decay 증가가 gap과 overfit_score를 거의 낮추지 못한다는 증거이고, run036-run039는 learning_rate를 낮추면 과적합은 줄지만 validation의 최저점도 일부 포기한다는 증거다. 다음으로 해석 가능한 축은 구조나 activation을 바꾸지 않는 dropout 강도다. drop_rate=0.12는 기존 0.10에서 작은 이동이라 MPS 회차를 길게 점유하지 않으면서, 0.0003 학습률의 낮은 validation loss와 0.000275 계열의 안정성 사이의 결합 가능성을 볼 수 있다.
- 바꾼 변수: `{"drop_rate": 0.12}`
- 기대 결과: 성공 기준은 run034 대비 final_generalization_gap이 0.0475보다 낮아지고 overfit_score가 0.12 이하 또는 fit_status=generalizing으로 회복되면서, final_val_loss가 5.565 이하에 머무는 것이다. final_val_loss가 5.57 이상으로 악화되면 dropout 증가가 under-training을 만든 것으로 본다. gap과 overfit_score가 거의 그대로면 seed=134의 80-step 문제는 dropout 강도보다 optimization 또는 데이터 window 쪽 원인일 가능성이 높다.
- 실제 결과: final_val_loss=5.554825623830159, gap=0.04473074277241995, overfit_score=0.13999509811401278, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0447, overfit_score=0.1400. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 drop_rate=0.12가 validation을 유지하며 overfit_score를 낮추면 seed=151 또는 seed=202에서 같은 설정을 반복해 regularization 이득이 seed 전반에 해롭지 않은지 확인한다. 이후 lr=0.0003/drop_rate=0.12 계열과 lr=0.000275/drop_rate=0.10 계열을 평균 validation, gap, overfit_score 기준으로 비교한다. / 과적합 시 drop_rate=0.12에서도 overfit_risk가 유지되면 dropout만으로는 seed=134의 80-step 과적합을 막기 어렵다고 보고, lr=0.000275를 안정 기본 후보로 유지하거나 max_steps=60을 안전 기준으로 되돌린다. 다음 후보는 drop_rate=0.12를 lr=0.000275와 결합하는 보수적 실험 또는 max_steps=70 중간 길이 테스트가 된다.

## run 041 - 2026-06-02T22:19:01+00:00

- 보고서: `docs/train/runs/run_041.md`
- 이번 가설: max_steps=70 seed=134 학습 길이 경계 테스트: run030의 max_steps=60은 seed=134에서 generalizing을 유지했지만 final_val_loss=5.588833으로 아직 덜 학습된 신호가 있었고, run034의 max_steps=80은 final_val_loss=5.554664까지 좋아졌지만 gap=0.047536, overfit_score=0.148410의 overfit_risk가 되었다. run040의 drop_rate=0.12는 validation을 거의 유지하며 status를 generalizing으로 되돌렸지만 gap과 overfit_score 감소폭은 작았다. 따라서 dropout을 더 건드리기 전에 max_steps만 70으로 줄이면, 80-step의 낮은 validation 이득을 상당 부분 유지하면서 seed=134의 과적합 신호를 줄일 수 있는지 확인한다.
- 근거: 최근 증거를 종합하면 context_length=48 + quick_gelu + sdpa + tie_embeddings=True + ffn_dropout_position=none은 유지할 가치가 높다. 과적합 완화 축에서는 weight_decay=0.02가 거의 무효였고, drop_rate=0.12는 약한 개선만 만들었다. learning_rate=0.000275는 세 seed에서 안정적이지만 best validation보다 조금 높다. 남은 핵심 질문은 learning_rate=0.0003 자체가 나쁜 것이 아니라 80 step까지 밀었을 때 seed=134에서 train 쪽 개선이 과하게 진행되는지다. max_steps=70은 구조와 함수 교체 없이 학습 길이만 조절하는 단일축이며, MPS balanced 장비에서 짧게 확인할 수 있다.
- 바꾼 변수: `{"max_steps": 70}`
- 기대 결과: 성공 기준은 run030(max_steps=60)보다 final_val_loss가 낮고, run034(max_steps=80)보다 final_generalization_gap과 overfit_score가 낮아지는 것이다. 구체적으로 final_val_loss가 5.565 이하이고 overfit_score가 0.12 이하 또는 gap이 0.04 이하이면 max_steps=70을 seed=134의 균형점 후보로 본다. final_val_loss가 5.58 이상이면 70 step도 under-training에 가깝고, gap이 0.047 전후로 유지되면 학습 길이만으로는 seed=134 과적합을 충분히 조절하지 못한 것으로 본다.
- 실제 결과: final_val_loss=5.569135665893555, gap=0.03515481948852539, overfit_score=0.1112673282623291, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0352, overfit_score=0.1113. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 max_steps=70이 validation과 overfit_score의 균형을 만들면 seed=151 또는 seed=202에서 같은 설정을 반복해 70-step이 평균적으로 안정적인지 확인한다. 성공하면 lr=0.0003/max_steps=70 계열, lr=0.000275/max_steps=80 계열, run033의 lr=0.0003/max_steps=80 seed=202 계열을 seed 평균 기준으로 비교한다. / 과적합 시 max_steps=70에서도 gap과 overfit_score가 크게 유지되면 seed=134에서는 learning_rate=0.0003 자체가 빠른 학습으로 train 편향을 만든다고 보고 lr=0.000275를 안정 기본 후보로 유지한다. 다음에는 lr=0.000275와 drop_rate=0.12의 결합 또는 max_steps=60 seed 반복을 선택한다.

## run 042 - 2026-06-02T22:24:10+00:00

- 보고서: `docs/train/runs/run_042.md`
- 이번 가설: learning_rate=0.000275 + drop_rate=0.12 seed=134 결합 테스트: run037은 learning_rate=0.000275에서 run034의 overfit_risk를 피하며 final_val_loss=5.563291, gap=0.039052, overfit_score=0.122958을 만들었고, run040은 learning_rate=0.0003에서 drop_rate=0.12를 적용해 validation을 거의 유지했지만 overfit_score 감소폭은 작았다. 따라서 안정 learning rate 계열인 0.000275 위에서 drop_rate만 0.10에서 0.12로 올리면, validation 손실을 크게 늘리지 않으면서 overfit_score를 0.10 이하로 낮출 수 있는지 확인한다.
- 근거: 최근 run041의 max_steps=70은 gap과 overfit_score를 줄였지만 final_val_loss가 5.569136으로 악화되어, 학습 길이 축만으로는 low-loss와 low-overfit의 균형이 충분하지 않았다. run035의 weight_decay 증가는 무효에 가까웠고, run040의 dropout 증가는 lr=0.0003 조건에서만 약한 개선을 보였다. 반면 learning_rate=0.000275는 seed=134/151/202에서 모두 generalizing을 유지해 가장 안정적인 optimization 후보로 볼 수 있다. 이번 실험은 run037을 기준으로 drop_rate만 바꾸는 단일축 결합 검증이며, 모델 구조와 함수 선택은 그대로 유지한다.
- 바꾼 변수: `{"drop_rate": 0.12}`
- 기대 결과: 성공 기준은 run037 대비 final_generalization_gap과 overfit_score가 낮아지고, final_val_loss가 5.57 이하에 머무는 것이다. 특히 overfit_score가 0.10 이하 또는 risk가 low로 내려가면 lr=0.000275에 약한 dropout을 결합하는 방향이 의미 있다고 본다. final_val_loss가 5.58 이상이면 dropout 추가가 under-training을 만든 것으로 보고, gap이 거의 줄지 않으면 seed=134 과적합은 dropout보다 optimization 경로 또는 데이터 window 특성의 영향이 크다고 본다.
- 실제 결과: final_val_loss=5.563780466715495, gap=0.03705386320749948, overfit_score=0.11696445941925138, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0371, overfit_score=0.1170. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 같은 lr=0.000275 + drop_rate=0.12 설정을 seed=151 또는 seed=202에 반복해 평균적으로 validation을 유지하는지 확인한다. 세 seed에서 안정적이면 이 계열을 low-risk 기본 후보로 두고, run033의 low-val 후보와 평균 score 기준으로 비교한다. / 과적합 시 과적합이 유지되거나 validation이 악화되면 lr=0.000275/drop_rate=0.10을 안정 기본 후보로 유지하고, 다음에는 activation_name=gelu_exact 또는 norm_eps 같은 함수/수치 안정성 축을 다시 작은 단일축으로 탐색한다. 또는 seed=151에서 lr=0.0003/max_steps=80을 확인해 seed 평균 비교를 보완한다.

## run 043 - 2026-06-02T22:29:08+00:00

- 보고서: `docs/train/runs/run_043.md`
- 이번 가설: learning_rate=0.0003 max_steps=80 seed=151 평균 후보 보완 테스트: run033은 seed=202에서 final_val_loss=5.553315, gap=0.010401, overfit_score=0.050397로 현재 best를 만들었고, run034는 seed=134에서 비슷한 validation loss를 얻었지만 overfit_risk가 되었다. seed=151은 아직 learning_rate=0.0003, max_steps=80 조합으로 직접 확인하지 않았다. 같은 설정을 seed=151에 적용하면 0.0003/80-step 계열이 평균적으로 낮은 validation을 만드는지, 아니면 seed에 따라 과적합 위험이 너무 큰지 판단할 수 있다.
- 근거: 최근 run040-run042는 dropout이나 max_steps=70, learning_rate=0.000275와 dropout 결합이 과적합 신호를 조금 낮추지만 best validation을 넘지는 못한다는 것을 보여줬다. 따라서 더 많은 regularization을 누르기보다, 먼저 최고 validation 계열의 seed 평균 근거를 보완하는 것이 다음 의사결정에 더 중요하다. seed=151은 run031(max_steps=60, lr=0.0003)에서 final_val_loss=5.595870, run039(max_steps=80, lr=0.000275)에서 final_val_loss=5.561801을 보여서, 80-step과 0.0003 학습률의 추가 이득과 과적합 비용을 동시에 관찰하기 좋은 남은 seed다.
- 바꾼 변수: `{"drop_rate": 0.1, "learning_rate": 0.0003, "seed": 151}`
- 기대 결과: 성공 기준은 seed=151에서도 final_val_loss가 run039의 5.561801보다 낮거나 비슷하고, overfit_score가 0.12 이하 또는 fit_status=generalizing을 유지하는 것이다. final_val_loss가 5.55대까지 내려가고 gap이 0.04 이하이면 0.0003/80-step은 평균 후보로 계속 유지한다. validation은 좋아지지만 overfit_score가 0.14 이상이면 0.0003은 low-loss 후보지만 seed 민감한 과적합 축으로 분류한다.
- 실제 결과: final_val_loss=5.553295294443767, gap=0.03364876906077097, overfit_score=0.09724970658620347, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0336, overfit_score=0.0972. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 lr=0.0003/max_steps=80의 세 seed 결과를 평균 validation, gap, overfit_score로 정리하고, best run033의 seed 특이성 여부를 판단한다. 이후에는 activation_name=gelu_exact 또는 norm_eps 같은 함수/수치 안정성 축을 이 best 계열 위에서 작게 테스트한다. / 과적합 시 과적합이 커지면 lr=0.0003/max_steps=80은 seed 202에서는 best지만 평균 위험이 큰 후보로 본다. 다음에는 lr=0.000275/max_steps=80을 안정 기본 후보로 유지하거나, lr=0.0003에 대해서는 max_steps=70 또는 drop_rate=0.12를 seed별로 제한적으로 비교한다.

## run 044 - 2026-06-02T22:34:06+00:00

- 보고서: `docs/train/runs/run_044.md`
- 이번 가설: gelu_exact activation 단일축 재검증: run043에서 learning_rate=0.0003, max_steps=80, seed=151은 final_val_loss=5.553295로 run033과 거의 같은 최상위 validation을 만들었지만 gap=0.033649, overfit_score=0.097250으로 low-risk까지는 내려가지 않았다. 같은 학습 조건에서 activation_name만 quick_gelu에서 gelu_exact로 바꾸면, GELU 근사 대신 정확한 GELU 곡선이 train 쪽 과도한 sharpness를 조금 줄여 validation과 overfit_score 균형을 개선할 수 있는지 확인한다.
- 근거: 최근 실험은 lr=0.0003/max_steps=80 계열이 seed=151과 seed=202에서 강한 validation을 만들고, seed=134에서만 과적합 위험이 커진다는 것을 보여줬다. dropout, weight_decay, max_steps=70, lr=0.000275 조합은 과적합 신호를 조금 낮췄지만 best validation을 넘지 못했다. 따라서 이제 regularization을 더 누르기보다 구조를 바꾸지 않는 함수 교체 축으로 넘어가는 것이 자연스럽다. gelu_exact는 이전 40-step 조건에서 quick_gelu와 거의 동등했지만, 현재처럼 80-step 저손실 구간에서는 근사 차이가 gap과 overfit_score에 다르게 나타날 수 있다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: 성공 기준은 run043 대비 final_val_loss가 5.56 이하를 유지하면서 final_generalization_gap 또는 overfit_score가 낮아지는 것이다. overfit_score가 0.08 이하로 내려가면 gelu_exact가 best 계열의 안정화 후보가 된다. final_val_loss가 비슷하지만 tokens_per_sec가 크게 나빠지면 quick_gelu를 기본 activation으로 유지한다. validation이 5.57 이상으로 악화되면 activation 변경보다 quick_gelu가 현재 학습 조건에 더 적합하다고 본다.
- 실제 결과: final_val_loss=5.553210894266765, gap=0.03350428740183542, overfit_score=0.09682873884836862, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0335, overfit_score=0.0968. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 같은 gelu_exact 설정을 seed=202 또는 seed=134에 반복해 activation 효과가 seed에 강건한지 확인한다. 특히 seed=134에서 overfit_score를 낮추는지 보면 quick_gelu의 빠른 학습 성향이 과적합에 기여했는지 해석할 수 있다. / 과적합 시 gelu_exact에서도 gap이 커지거나 validation이 악화되면 quick_gelu를 유지하고, 다음에는 norm_eps=1e-6 또는 norm_eps=1e-4처럼 LayerNorm 수치 안정성 축을 작은 단일축으로 테스트한다. activation 계열 확장은 swiglu/geglu처럼 parameter_count를 바꾸는 후보보다 mish/silu 같은 동일 parameter_count 후보를 먼저 본다.

## run 045 - 2026-06-02T22:39:47+00:00

- 보고서: `docs/train/runs/run_045.md`
- 이번 가설: best seed=202에서 gelu_exact activation 단일축 검증: run044에서 seed=151은 quick_gelu 대비 gelu_exact가 final_val_loss와 overfit_score를 아주 작게 개선했다. 현재 best인 run033은 seed=202, learning_rate=0.0003, max_steps=80, quick_gelu 조건에서 final_val_loss=5.553315, gap=0.010401, overfit_score=0.050397을 만들었다. 같은 best 조건에서 activation_name만 gelu_exact로 바꾸면, seed=151에서 보인 미세한 개선이 seed=202에서도 재현되어 best 후보를 갱신할 수 있는지 확인한다.
- 근거: 최근 결과는 lr=0.0003/max_steps=80 계열이 seed=151과 seed=202에서 가장 낮은 validation을 만들고, seed=134에서만 과적합 위험이 커진다는 패턴을 보인다. run044의 gelu_exact는 parameter_count와 구조를 바꾸지 않고 run043보다 final_val_loss를 0.000084 낮추고 overfit_score도 아주 작게 낮췄다. 이 차이는 작지만 실험 조건이 동일하므로 해석 가치가 있다. 이번에는 best run033과 완전히 같은 seed=202 조건에서 activation만 교체해, gelu_exact가 실제 best 계열의 기본 activation 후보가 될 수 있는지 검증한다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: 성공 기준은 run033 대비 final_val_loss가 같거나 더 낮고, overfit_score가 0.05 이하 또는 gap이 0.011 이하로 유지되는 것이다. final_val_loss가 5.5530 이하로 내려가면 best 후보 갱신 가능성이 높다. final_val_loss가 비슷하지만 tokens_per_sec가 크게 느려지면 quick_gelu의 실용성이 더 높다고 본다. validation이나 gap이 악화되면 seed=151의 gelu_exact 개선은 작은 노이즈 또는 seed-specific 효과로 해석한다.
- 실제 결과: final_val_loss=5.553322792053223, gap=0.010373592376708984, overfit_score=0.0502632459004726, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0104, overfit_score=0.0503. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 gelu_exact를 seed=134에 적용해 과적합이 컸던 seed에서도 gap과 overfit_score를 낮추는지 확인한다. seed=202와 seed=151 모두에서 개선되면 gelu_exact를 best 계열 activation 후보로 올리고, 다음에는 norm_eps 단일축을 gelu_exact 위에서 테스트한다. / 과적합 시 gelu_exact가 seed=202에서 validation이나 overfit_score를 악화시키면 quick_gelu를 기본 activation으로 유지한다. 다음에는 activation 대신 norm_eps=1e-6 또는 norm_eps=1e-4처럼 LayerNorm 수치 안정성 축을 best run033 조건에서 테스트한다.

## run 046 - 2026-06-02T22:44:13+00:00

- 보고서: `docs/train/runs/run_046.md`
- 이번 가설: seed=134 과적합 구간에서 gelu_exact activation 단일축 검증: run034는 learning_rate=0.0003, max_steps=80, seed=134, quick_gelu 조건에서 final_val_loss=5.554664까지 낮아졌지만 gap=0.047536, overfit_score=0.148410으로 overfit_risk가 되었다. 반면 run044(seed=151)와 run045(seed=202)에서는 같은 저손실 계열에서 gelu_exact가 quick_gelu 대비 validation 또는 overfit_score를 아주 작게 개선했다. 따라서 run034와 동일한 seed=134 조건에서 activation_name만 gelu_exact로 바꾸면 낮은 validation은 유지하면서 train 쪽 과도 적합 신호를 줄일 수 있는지 확인한다.
- 근거: 현재 best는 run045이지만, 그 개선은 seed=202에서 gap이 이미 낮은 조건의 미세 개선이다. 연구적으로 더 중요한 질문은 seed=134처럼 0.0003/80-step 계열이 overfit_risk를 보이는 경우에도 gelu_exact가 안정화 효과를 내는지다. weight_decay=0.02, drop_rate=0.12, max_steps=70, learning_rate=0.000275는 모두 gap을 일부 낮췄지만 best validation을 명확히 넘지 못했다. activation 교체는 구조 순서를 바꾸지 않고 parameter_count도 유지하므로, 과적합 원인이 quick_gelu의 근사/스케일 특성과 관련 있는지 해석하기 좋은 단일 함수 교체 실험이다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: 성공 기준은 run034 대비 final_val_loss가 5.56 이하를 유지하면서 final_generalization_gap이 0.0475보다 낮아지고 overfit_score가 0.14 이하 또는 fit_status=generalizing으로 내려오는 것이다. final_val_loss가 거의 같고 gap/overfit_score만 낮아지면 gelu_exact를 best 저손실 계열의 기본 activation 후보로 올린다. validation은 유지되지만 gap이 그대로면 activation 차이는 seed=134 과적합 완화에는 충분하지 않다고 본다. validation이 5.57 이상으로 악화되면 gelu_exact 개선은 seed=151/202의 미세 노이즈로 분류한다.
- 실제 결과: final_val_loss=5.554612954457601, gap=0.04752751191457083, overfit_score=0.14835798740386874, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0475, overfit_score=0.1484. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 성공하면 gelu_exact가 세 seed에서 모두 저손실 계열에 해롭지 않고 seed=134 과적합도 완화한다는 근거가 생긴다. 다음 실험은 gelu_exact를 유지한 채 norm_eps=1e-6 또는 1e-4를 단일축으로 테스트해 LayerNorm 수치 안정성이 gap을 더 낮추는지 본다. / 과적합 시 overfit_risk가 유지되면 quick_gelu와 gelu_exact의 차이는 과적합 원인이 아니라고 보고 activation 계열 확장을 잠시 멈춘다. 다음에는 seed=134에서 learning_rate=0.000275 또는 max_steps=70 계열을 기준으로 norm_eps나 ffn_dropout_position 같은 수치/드롭아웃 위치 축을 더 보수적으로 확인한다.

## run 047 - 2026-06-02T22:49:08+00:00

- 보고서: `docs/train/runs/run_047.md`
- 이번 가설: seed=134 저손실 과적합 구간에서 FFN dropout 위치 단일축 검증: run034와 run046은 learning_rate=0.0003, max_steps=80, seed=134 조건에서 final_val_loss를 5.55대까지 낮췄지만 gap≈0.0475, overfit_score≈0.148의 overfit_risk를 반복했다. weight_decay 증가와 gelu_exact 교체는 거의 효과가 없었고, drop_rate=0.12는 약하게만 완화했다. 따라서 기존 구조 순서는 유지한 채 ffn_dropout_position만 none에서 after_activation으로 바꾸면, FFN hidden activation 직후에 regularization이 걸려 train 쪽 과도 적합을 줄이면서 낮은 validation loss를 유지할 수 있는지 확인한다.
- 근거: 최근 evidence는 seed=134의 문제를 activation 근사 차이나 weight_decay로 설명하기 어렵다는 쪽으로 모였다. run040은 dropout 강도를 0.12로 올렸을 때 final_val_loss를 유지하면서 overfit_score를 0.148에서 0.140으로 조금 낮췄지만 충분하지 않았다. ffn_dropout_position=after_activation은 drop_rate 자체를 더 키우지 않고 FFN의 확장 hidden 표현에 dropout을 적용하므로, output 직후나 none보다 train memorization을 다르게 억제할 수 있다. parameter_count와 Transformer block 순서는 그대로이며, 실험 가능한 작은 함수/위치 교체 축이라 해석 가능성이 높다.
- 바꾼 변수: `{"ffn_dropout_position": "after_activation"}`
- 기대 결과: 성공 기준은 run034 대비 final_val_loss가 5.565 이하를 유지하면서 final_generalization_gap이 0.04 이하 또는 overfit_score가 0.12 이하로 내려가는 것이다. final_val_loss가 5.55대에 머물고 fit_status가 generalizing으로 바뀌면 FFN 내부 dropout 위치가 seed=134의 과적합 완화에 의미 있다고 본다. gap이 거의 줄지 않으면 FFN dropout 위치보다 learning_rate 또는 학습 길이가 핵심 원인이다. validation이 5.58 이상으로 악화되면 after_activation dropout은 under-training을 만든 것으로 본다.
- 실제 결과: final_val_loss=5.55439821879069, gap=0.04678706328074167, overfit_score=0.14616405963897794, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0468, overfit_score=0.1462. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 같은 ffn_dropout_position=after_activation 설정을 seed=151 또는 seed=202에 반복해 best 저손실 계열을 해치지 않는지 확인한다. 세 seed에서 안정적이면 after_activation을 low-overfit 후보로 두고, 이후 norm_eps 단일축을 이 조건 위에서 테스트한다. / 과적합 시 overfit_risk가 유지되면 dropout 위치만으로는 seed=134의 train 편향을 막기 어렵다고 보고, learning_rate=0.000275 또는 max_steps=70을 핵심 안정화 축으로 채택한다. 다음에는 lr=0.000275 계열에서 norm_eps=1e-6 또는 ffn_dropout_position=after_activation을 보수적으로 결합해 gap을 낮추는지 확인한다.

## run 048 - 2026-06-02T22:54:19+00:00

- 보고서: `docs/train/runs/run_048.md`
- 이번 가설: learning_rate=0.000275 안정화 계열에서 FFN dropout 위치 결합 검증: run037은 seed=134에서 learning_rate를 0.000275로 낮추자 run034의 overfit_risk를 generalizing으로 되돌렸지만 gap=0.039052, overfit_score=0.122958로 아직 medium-risk에 가까웠다. run047은 높은 learning_rate=0.0003 조건에서 ffn_dropout_position=after_activation이 validation을 유지하며 overfit_score를 0.148에서 0.146으로 아주 작게 낮췄다. 따라서 run037과 같은 안정 learning_rate 조건에서 ffn_dropout_position만 after_activation으로 바꾸면, validation 손실을 크게 늘리지 않고 gap과 overfit_score를 더 낮출 수 있는지 확인한다.
- 근거: 최근 결과는 seed=134 과적합의 주요 원인이 activation이나 weight_decay보다 optimization 속도에 가깝다는 쪽으로 모였다. learning_rate=0.000275는 세 seed에서 generalizing을 유지한 안정 축이고, after_activation dropout 위치는 단독으로는 약했지만 train memorization을 아주 조금 완화했다. 이번 실험은 run037을 기준으로 dropout 위치만 바꾸는 단일축 결합 검증이다. 구조 순서, parameter_count, context_length, attention_impl, activation은 유지되므로 해석 가능성이 높고, MPS balanced 장비에서도 80 step이 1초 안팎이라 자동 루프 점유가 작다.
- 바꾼 변수: `{"ffn_dropout_position": "after_activation"}`
- 기대 결과: 성공 기준은 run037 대비 final_generalization_gap이 0.039 이하로 낮아지고 overfit_score가 0.115 이하로 내려가며, final_val_loss가 5.57 이하에 머무는 것이다. 특히 run042의 lr=0.000275 + drop_rate=0.12 결과인 overfit_score=0.116964보다 낮아지면 dropout 강도 증가보다 FFN 내부 위치 변경이 더 효율적인 regularization일 수 있다. final_val_loss가 5.58 이상으로 악화되면 after_activation 위치가 안정 lr 위에서는 under-training을 만든 것으로 본다. gap과 overfit_score가 거의 그대로면 다음에는 norm_eps 축으로 이동한다.
- 실제 결과: final_val_loss=5.563193480173747, gap=0.03860954443613718, overfit_score=0.12163150310516446, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0386, overfit_score=0.1216. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 ffn_dropout_position=after_activation을 lr=0.000275 안정 후보에 붙인 뒤 seed=151 또는 seed=202로 반복해 평균적으로 validation 손실 없이 gap을 낮추는지 확인한다. 세 seed에서 안정되면 이 계열을 low-risk 기본 후보로 두고, 이후 norm_eps=1e-6 또는 1e-4를 단일축으로 테스트한다. / 과적합 시 overfit_score가 유지되거나 validation이 악화되면 ffn_dropout_position 축은 seed=134 문제의 핵심 해결책이 아니라고 보고 중단한다. 다음 실험은 run037 또는 run042 조건에서 norm_eps=1e-6/1e-4를 비교하거나, max_steps=70 계열을 seed=151/202로 반복해 학습 길이 안정성을 확인한다.

## run 049 - 2026-06-02T22:59:05+00:00

- 보고서: `docs/train/runs/run_049.md`
- 이번 가설: learning_rate=0.000275 + drop_rate=0.12 안정 조합에서 norm_eps=1e-4 단일축 검증: run042는 seed=134에서 learning_rate=0.000275와 drop_rate=0.12를 결합해 final_val_loss=5.563780, gap=0.037054, overfit_score=0.116964로 run037보다 과적합 신호를 낮췄다. run048의 ffn_dropout_position=after_activation은 validation은 약간 좋았지만 overfit_score가 0.121632로 run042보다 높았다. 따라서 run042 조건을 기준으로 norm_eps만 1e-5에서 1e-4로 키우면 LayerNorm 분모 안정화가 train 쪽 sharp fitting을 조금 완화해 validation을 유지하면서 gap과 overfit_score를 더 낮출 수 있는지 확인한다.
- 근거: 최근 실험들은 seed=134의 과적합이 activation이나 weight_decay보다 optimization 속도와 regularization 위치/강도에 더 민감하다는 것을 보여줬다. drop_rate=0.12는 high learning_rate에서는 약했고, learning_rate=0.000275 위에서는 run042처럼 overfit_score를 0.116964까지 낮췄다. ffn_dropout_position은 단독 및 결합 모두 작게만 개선됐다. 이제 구조와 학습 길이를 유지하면서 남은 해석 가능한 단일축은 LayerNorm 수치 안정성이다. norm_eps=1e-4는 parameter_count와 Transformer 순서를 바꾸지 않고 normalization을 조금 더 부드럽게 만들어 seed=134의 train/val gap을 줄일 가능성이 있다.
- 바꾼 변수: `{"norm_eps": 0.0001}`
- 기대 결과: 성공 기준은 run042 대비 final_val_loss가 5.57 이하를 유지하고, final_generalization_gap이 0.037 이하 또는 overfit_score가 0.11 이하로 내려가는 것이다. validation이 거의 같고 overfit_score만 낮아지면 norm_eps=1e-4를 seed=134 안정화 후보로 본다. final_val_loss가 5.58 이상으로 악화되면 큰 eps가 normalization을 과도하게 둔화해 under-training을 만든 것으로 본다. gap과 overfit_score가 거의 같으면 norm_eps는 현 설정에서 의미 있는 축이 아니며 다음에는 max_steps=70 또는 seed 반복으로 이동한다.
- 실제 결과: final_val_loss=5.5637868245442705, gap=0.037049969037373565, overfit_score=0.11693807442982962, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0370, overfit_score=0.1169. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 norm_eps=1e-4를 learning_rate=0.000275 + drop_rate=0.12 계열에 유지한 채 seed=151 또는 seed=202로 반복해 평균적으로 validation 손실 없이 overfit_score를 낮추는지 확인한다. 안정되면 이 계열을 low-risk 후보로 두고 best run45의 low-val 후보와 평균 score로 비교한다. / 과적합 시 overfit_score가 유지되거나 validation이 악화되면 norm_eps=1e-4는 포기하고 기본 1e-5로 되돌린다. 다음에는 norm_eps=1e-6처럼 반대 방향의 수치 안정성 축을 짧게 확인하거나, max_steps=70 계열을 seed=151/202로 반복해 학습 길이 안정성을 비교한다.

## run 050 - 2026-06-02T23:05:16+00:00

- 보고서: `docs/train/runs/run_050.md`
- 이번 가설: 현재 best run45 위에서 drop_rate=0.12 단일축 검증: run45는 seed=202, learning_rate=0.0003, max_steps=80, gelu_exact 조건에서 final_val_loss=5.553323, gap=0.010374, overfit_score=0.050263으로 현재 best다. 최근 seed=134 실험들은 drop_rate=0.12가 validation을 크게 해치지 않으면서 overfit_score를 아주 약하게 낮추는 경향을 보였다. 따라서 run45와 동일한 조건에서 drop_rate만 0.10에서 0.12로 올리면 낮은 validation을 유지하면서 gap과 overfit_score를 더 낮춰 best 후보를 갱신할 수 있는지 확인한다.
- 근거: 최근 run046-run049는 seed=134의 과적합을 완화하려고 activation, FFN dropout 위치, norm_eps를 확인했지만 개선 폭은 작았다. 반면 현재 best는 seed=202에서 이미 low-risk라서 작은 regularization 조정이 pure validation을 크게 훼손하지 않으면 score 개선 가능성이 있다. drop_rate는 구조 순서를 바꾸지 않고 parameter_count도 유지하며, run040에서 high learning_rate 조건의 validation을 거의 유지한 축이다. 이번 실험은 seed=134 안정화 탐색에서 잠시 벗어나 현재 best를 직접 개선할 수 있는지 보는 exploitation 성격의 단일축 테스트다.
- 바꾼 변수: `{"drop_rate": 0.12}`
- 기대 결과: 성공 기준은 run45 대비 final_val_loss가 5.56 이하를 유지하고, final_generalization_gap이 0.010 이하 또는 overfit_score가 0.05 이하로 낮아지는 것이다. final_val_loss가 거의 같고 overfit_score가 낮아지면 drop_rate=0.12를 best 계열의 후보로 올린다. final_val_loss가 5.565 이상으로 악화되면 seed=202 best 조건에서는 dropout 증가가 under-training 또는 과도 regularization을 만든 것으로 본다. gap과 overfit_score가 거의 같으면 drop_rate 축은 best 개선에 유효하지 않다고 보고 seed 반복 검증으로 이동한다.
- 실제 결과: final_val_loss=5.553958892822266, gap=0.007346510887145996, overfit_score=0.04118200143178363, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0073, overfit_score=0.0412. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 drop_rate=0.12 + gelu_exact + learning_rate=0.0003 계열을 seed=151에 반복해 best 개선이 seed=202 특이 효과인지 확인한다. seed=151에서도 validation과 overfit_score가 안정되면 이 계열을 저손실 후보로 유지하고, seed=134에는 learning_rate=0.000275 안정화 계열을 별도로 둔다. / 과적합 시 overfit_score가 높아지거나 validation이 악화되면 run45의 drop_rate=0.10을 유지한다. 다음에는 현재 best 계열의 seed 평균을 더 명확히 하기 위해 seed=151 또는 seed=134에서 drop_rate를 더 건드리지 않고 max_steps=70/80 경계 또는 seed 반복을 진행한다.

## run 051 - 2026-06-02T23:09:04+00:00

- 보고서: `docs/train/runs/run_051.md`
- 이번 가설: 새 best run50 조합의 seed=151 재현성 검증: run50은 seed=202, learning_rate=0.0003, max_steps=80, gelu_exact, drop_rate=0.12 조건에서 final_val_loss=5.553959, gap=0.007347, overfit_score=0.041182로 새 best가 되었다. 같은 seed=151 계열의 run44는 gelu_exact와 drop_rate=0.10에서 final_val_loss=5.553211로 validation은 아주 좋았지만 gap=0.033504, overfit_score=0.096829로 low-risk까지는 아니었다. 따라서 run50과 동일한 저손실 조합에서 seed만 151로 바꾸면, drop_rate=0.12가 seed=202 특이 개선이 아니라 seed=151에서도 validation을 유지하면서 gap과 overfit_score를 낮추는지 확인할 수 있다.
- 근거: 최근 시각 추세와 leaderboard는 learning_rate=0.0003/max_steps=80 계열이 validation loss 최저권을 안정적으로 만들지만 seed에 따라 과적합 신호가 크게 달라진다는 것을 보여준다. run50은 run45 대비 validation은 0.000636 높아졌지만 gap과 overfit_score를 더 낮춰 overfit-aware score에서 best가 되었다. 이제 새 activation이나 capacity 축으로 이동하기 전에, 같은 조합을 seed=151에 반복해 평균 후보로 삼을 수 있는지 확인하는 것이 해석 가능성이 높다. 이 실험은 Transformer 구조와 함수 조합을 유지하고 seed만 바꾸는 재현성 테스트라, 하드웨어 점유도 짧고 결과 해석도 명확하다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 seed=151의 기존 gelu_exact run44 대비 final_val_loss가 5.56 이하를 유지하면서 final_generalization_gap이 0.0335보다 낮아지고 overfit_score가 0.09 이하로 내려가는 것이다. final_val_loss가 5.553 전후로 유지되고 overfit_score가 낮아지면 drop_rate=0.12를 best 계열의 평균 후보로 본다. validation이 5.565 이상으로 악화되면 seed=151에서는 dropout 증가가 과도 regularization을 만든 것으로 보고 run44의 drop_rate=0.10을 더 좋은 seed=151 기준으로 유지한다.
- 실제 결과: final_val_loss=5.553611596425374, gap=0.0301429828008013, overfit_score=0.08674482504526626, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0301, overfit_score=0.0867. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed=151에서도 drop_rate=0.12가 validation을 유지하며 overfit_score를 낮추면, 다음에는 seed=134에서 같은 gelu_exact/drop_rate=0.12 조합을 확인해 가장 어려운 seed에서도 high learning rate 계열의 과적합을 완화하는지 본다. 세 seed 평균이 안정되면 learning_rate=0.0003/max_steps=80/drop_rate=0.12/gelu_exact를 low-loss 기본 후보로 승격한다. / 과적합 시 seed=151에서 gap이나 overfit_score가 커지거나 validation이 악화되면 run50은 seed=202 특화 개선으로 분류한다. 다음에는 seed=151은 run44 조건(drop_rate=0.10)을 유지하고, seed=134는 learning_rate=0.000275/drop_rate=0.12 안정화 계열을 기준으로 seed 평균 또는 max_steps 경계 실험을 이어간다.

## run 052 - 2026-06-02T23:14:00+00:00

- 보고서: `docs/train/runs/run_052.md`
- 이번 가설: seed=134 저손실 과적합 구간에서 gelu_exact + drop_rate=0.12 결합 검증: run046은 seed=134, learning_rate=0.0003, max_steps=80, gelu_exact, drop_rate=0.10 조건에서 final_val_loss=5.554613으로 저손실을 유지했지만 gap=0.047528, overfit_score=0.148358로 overfit_risk였다. run050과 run051은 같은 gelu_exact 저손실 계열에서 drop_rate=0.12가 seed=202와 seed=151의 overfit_score를 낮추는 방향을 보였다. 따라서 run046과 동일한 seed=134 조건에서 drop_rate만 0.12로 올리면, 어려운 seed에서도 validation을 유지하면서 gap과 overfit_score를 낮춰 이 조합을 세 seed 평균 후보로 승격할 수 있는지 확인한다.
- 근거: 최근 dashboard의 추세는 learning_rate=0.0003/max_steps=80 계열이 validation loss 최저권을 만들지만, seed=134에서 train 쪽 개선이 과하게 진행되어 overfit_score가 커지는 패턴을 보여준다. gelu_exact 단독은 seed=134의 과적합을 거의 줄이지 못했지만(run046), drop_rate=0.12는 seed=202에서 best를 만들고(run050), seed=151에서도 validation을 유지하면서 gap과 overfit_score를 낮췄다(run051). 이제 가장 어려운 seed=134에서 같은 regularization을 결합해 평균 후보인지, 아니면 seed134에는 learning_rate=0.000275 안정화 계열을 별도로 둬야 하는지 판단해야 한다. 구조 순서, attention 구현, FFN 형태, parameter_count는 유지하고 dropout 강도만 바꾸는 작은 실험이라 해석 가능성이 높다.
- 바꾼 변수: `{"drop_rate": 0.12}`
- 기대 결과: 성공 기준은 run046 대비 final_val_loss가 5.56 이하를 유지하고, final_generalization_gap이 0.0475보다 낮아지며, overfit_score가 0.14 이하 또는 fit_status가 generalizing으로 내려오는 것이다. 특히 overfit_score가 0.12 이하에 가까워지면 gelu_exact + drop_rate=0.12 조합이 seed134 과적합 완화에도 의미 있다고 본다. final_val_loss가 5.565 이상으로 악화되면 seed134에서는 dropout 증가가 저손실 이득을 훼손한 것으로 본다. gap과 overfit_score가 거의 그대로면 seed134 문제는 dropout/activation보다 optimization 속도에 가깝다고 보고 learning_rate=0.000275 계열을 별도 안정 후보로 유지한다.
- 실제 결과: final_val_loss=5.554793834686279, gap=0.044721126556396484, overfit_score=0.1399388313293457, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0447, overfit_score=0.1399. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 learning_rate=0.0003/max_steps=80/gelu_exact/drop_rate=0.12 조합을 세 seed의 저손실 평균 후보로 정리하고, 다음에는 seed별 평균 점수를 계산해 run050 계열과 learning_rate=0.000275 안정 계열을 비교한다. 추가 실험은 max_steps=90 같은 더 긴 학습이 아니라, seed 반복이나 context/stride 같은 데이터 window 축으로 이동한다. / 과적합 시 과적합이 유지되면 seed134에서는 high learning rate 80-step 계열의 train 편향이 핵심이라고 판단한다. 다음에는 seed134를 learning_rate=0.000275/drop_rate=0.12 또는 max_steps=70 계열로 두고, seed202/151의 low-loss 계열과 별도로 운영하는 하이브리드 기준을 세운다. function 교체 축은 잠시 멈추고 optimization과 학습 길이 경계 실험을 우선한다.

## run 053 - 2026-06-02T23:19:05+00:00

- 보고서: `docs/train/runs/run_053.md`
- 이번 가설: seed=134 gelu_exact + drop_rate=0.12 조건에서 learning_rate=0.000275 안정화 검증: run052는 seed=134, gelu_exact, drop_rate=0.12, learning_rate=0.0003, max_steps=80 조건에서 final_val_loss=5.554794로 저손실을 유지했지만 gap=0.044721, overfit_score=0.139939로 여전히 train 편향이 컸다. run037/run042는 learning_rate=0.000275가 seed134의 과적합 신호를 낮추는 가장 일관된 축임을 보여줬다. 따라서 run052와 동일한 함수/regularization 조합에서 learning_rate만 0.000275로 낮추면 validation 손실을 크게 늘리지 않으면서 gap과 overfit_score를 안정적으로 낮출 수 있는지 확인한다.
- 근거: 최근 결과를 종합하면 seed134의 문제는 gelu_exact 단독(run046)이나 dropout 증가(run052)만으로 충분히 해결되지 않았다. 반대로 learning_rate=0.000275 계열은 seed134에서 final_val_loss를 약간 희생하지만 gap과 overfit_score를 낮추는 안정화 효과가 반복적으로 관찰됐다. 이번 실험은 run052를 기준으로 learning_rate만 바꾸는 단일축 optimization 테스트라, drop_rate=0.12와 gelu_exact를 유지한 상태에서도 낮은 learning_rate가 핵심인지 분리할 수 있다. MPS balanced 장비에서 80 step은 약 1초 내외라 자동화 점유도 안전하다.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: 성공 기준은 run052 대비 final_generalization_gap이 0.04 이하로 내려가고 overfit_score가 0.12 이하로 낮아지며, final_val_loss가 5.57 이하에 머무는 것이다. run042와 비슷한 validation을 보이면서 gap이나 overfit_score가 더 낮아지면 seed134 안정 후보로 채택한다. final_val_loss가 5.58 이상이면 learning_rate 감소가 under-training을 만든 것으로 본다. gap과 overfit_score가 거의 줄지 않으면 seed134 안정화는 learning_rate 단독보다 max_steps 또는 데이터 window 축을 필요로 한다.
- 실제 결과: final_val_loss=5.563757101694743, gap=0.03705958525339792, overfit_score=0.11695420742035001, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0371, overfit_score=0.1170. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 seed134는 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact를 안정 경로로 두고, seed151/202의 low-loss 경로(run50/run51)와 분리한 하이브리드 전략을 문서화한다. 다음 실험은 seed151 또는 seed202에서 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact를 반복해 평균 안정성 점수를 비교하거나, context_length/stride 축으로 넘어간다. / 과적합 시 과적합이 유지되면 learning_rate만으로는 부족하다고 보고 max_steps=70과 learning_rate=0.000275의 결합을 seed134에서 확인한다. validation이 크게 나빠지면 learning_rate=0.0003 저손실 후보는 유지하되, seed134에는 early stop 또는 max_steps 경계 실험을 우선한다. function 교체 축은 추가하지 않는다.

## run 054 - 2026-06-02T23:24:18+00:00

- 보고서: `docs/train/runs/run_054.md`
- 이번 가설: seed=151에서 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact 안정 후보 검증: run051은 seed=151, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 final_val_loss=5.553612로 저손실을 유지했지만 gap=0.030143, overfit_score=0.086745로 medium risk였다. run053은 seed=134에서 learning_rate=0.000275가 validation을 일부 희생하더라도 gap과 overfit_score를 안정화한다는 것을 보였다. 따라서 seed=151에서도 run051과 동일한 함수/regularization 조합에서 learning_rate만 0.000275로 낮추면, validation 손실을 크게 키우지 않으면서 overfit_score를 더 낮춰 안정 평균 후보가 될 수 있는지 확인한다.
- 근거: 현재 best는 seed202의 run050이지만, seed134와 seed151은 같은 저손실 계열에서 gap이 더 크다. seed134에는 learning_rate=0.000275가 반복적으로 안정화 효과를 보였고, seed151도 기존 run039에서 learning_rate=0.000275가 final_val_loss=5.561801, overfit_score=0.068929로 꽤 안정적이었다. 이번 실험은 run051을 기준으로 learning_rate만 낮추는 단일축 테스트이며, gelu_exact와 drop_rate=0.12가 seed151에서 안정화와 validation 사이의 더 나은 균형을 만드는지 확인한다. 구조와 parameter_count는 그대로 유지한다.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: 성공 기준은 run051 대비 final_generalization_gap과 overfit_score가 낮아지고, final_val_loss가 5.57 이하에 머무는 것이다. 특히 overfit_score가 0.07 이하로 내려가면 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact를 seed151 안정 후보로 본다. final_val_loss가 5.58 이상이면 learning_rate 감소와 dropout 조합이 under-training을 만든 것으로 본다. validation이 run039와 비슷하고 overfit_score가 더 낮으면 안정 후보로 유지한다.
- 실제 결과: final_val_loss=5.562357584635417, gap=0.021705786387125947, overfit_score=0.061433235804240205, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0217, overfit_score=0.0614. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact를 seed134/151 안정 경로로 두고, 다음에는 seed202에도 같은 조합을 반복해 안정 경로의 세 seed 평균을 완성한다. 이후 run050/run051의 low-loss 경로와 안정 경로를 평균 score, pure validation, overfit_score 기준으로 비교한다. / 과적합 시 gap이나 overfit_score가 줄지 않으면 seed151은 run051의 low-loss 경로를 유지하고, 안정화는 seed134에만 적용하는 하이브리드 전략으로 둔다. validation이 크게 악화되면 learning_rate=0.000275와 drop_rate=0.12의 결합은 seed151에서 과도 regularization으로 판단하고, 다음에는 context_length/stride 같은 데이터 window 축을 본다.

## run 055 - 2026-06-02T23:29:48+00:00

- 보고서: `docs/train/runs/run_055.md`
- 이번 가설: seed=202에서 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact 안정 경로 검증: run050은 seed=202, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 현재 best(final_val_loss=5.553959, gap=0.007347, overfit_score=0.041182)를 만들었다. run053/run054는 seed134/151에서 learning_rate=0.000275가 validation을 일부 희생하더라도 gap과 overfit_score를 낮춰 안정화한다는 것을 보였다. 따라서 run050과 동일한 함수/regularization 조합에서 learning_rate만 0.000275로 낮추면, seed202에서도 validation 손실 증가를 작게 유지하면서 overfit_score를 더 낮춰 안정 경로의 세 seed 평균 비교를 완성할 수 있는지 확인한다.
- 근거: 현재 leaderboard는 learning_rate=0.0003 계열이 pure validation loss 최저권을 만들고, learning_rate=0.000275 계열이 seed134/151에서 과적합 지표를 낮추는 경향을 보여준다. 다만 seed202의 best run050은 이미 gap이 매우 낮아서 낮은 learning_rate가 실제로 의미 있는 안정성 개선인지, 아니면 validation만 희생하는지 아직 분리되지 않았다. 이번 실험은 run050을 기준으로 learning_rate만 바꾸는 단일축 optimization 테스트라 해석 가능성이 높고, Transformer 구조와 activation/attention/FFN 형태는 모두 유지한다. MPS 환경에서 max_steps=80은 짧은 회차로 끝나므로 자동 루프 점유도 안전하다.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: 성공 기준은 run050 대비 final_generalization_gap이나 overfit_score가 낮아지고, final_val_loss가 5.57 이하에 머무는 것이다. 특히 final_val_loss가 5.56대 초반 이하이고 overfit_score가 0.03 이하로 내려가면 seed202에서도 안정 경로가 의미 있다고 본다. final_val_loss가 5.575 이상으로 상승하면 learning_rate 감소는 seed202 best 경로에서 과도한 under-training 비용으로 판단한다.
- 실제 결과: final_val_loss=5.564117431640625, gap=0.0014352798461914062, overfit_score=0.023448308308919863, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0014, overfit_score=0.0234. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 learning_rate=0.000275 + drop_rate=0.12 + gelu_exact 안정 경로의 seed134/151/202 평균을 run050/run051/run052의 저손실 경로 평균과 비교한다. 평균 validation 손실 차이가 작고 overfit_score 평균이 낮으면 안정 경로를 기본 후보로 승격하고, 다음에는 context_length/stride 같은 데이터 window 축을 실험한다. / 과적합 시 과적합 신호가 줄지 않거나 validation만 악화되면 seed202는 run050의 learning_rate=0.0003 저손실 경로를 유지한다. 이후 seed134/151에는 안정 경로, seed202에는 저손실 경로를 쓰는 하이브리드 기준을 문서화하고, 다음 실험은 max_steps=70 또는 context/stride 축으로 이동한다.

## run 056 - 2026-06-02T23:34:19+00:00

- 보고서: `docs/train/runs/run_056.md`
- 이번 가설: seed=134 저손실 경로에서 stride=24 데이터 window 축 검증: run050/run051/run052의 learning_rate=0.0003 + drop_rate=0.12 + gelu_exact 저손실 경로는 세 seed 평균 validation이 안정 경로보다 좋았지만, seed=134 run052는 gap=0.044721, overfit_score=0.139939로 train 편향이 여전히 컸다. 반대로 learning_rate=0.000275 안정 경로는 overfit_score를 낮췄지만 validation을 약 0.009 올렸다. 따라서 run052와 동일한 모델/함수/학습률 조건에서 stride만 null에서 24로 줄여 overlapping training windows를 늘리면, 구조를 바꾸지 않고 데이터 노출 다양성을 늘려 validation 손실을 유지하면서 seed134의 gap과 overfit_score를 낮출 수 있는지 확인한다.
- 근거: 지금까지 activation, dropout 강도, norm_eps, learning_rate를 확인한 결과, low-loss를 만드는 핵심은 learning_rate=0.0003 + max_steps=80 계열이고 과적합 위험은 특히 seed134에서 커졌다. learning_rate를 낮추면 overfit_score는 낮아지지만 validation이 확실히 악화된다. 다음으로 해석 가능한 작은 축은 Transformer 구조가 아니라 데이터 window 구성이다. stride=24는 context_length=48과 모델 크기를 유지하면서 학습 샘플의 시작점을 더 촘촘히 만들어 작은 corpus에서 같은 80 step 동안 더 다양한 국소 문맥을 보게 한다. MPS balanced 하드웨어에서는 batch_size=8, max_steps=80이 계속 짧은 회차이므로 안전하다.
- 바꾼 변수: `{"stride": 24}`
- 기대 결과: 성공 기준은 run052 대비 final_val_loss가 5.56 이하를 유지하면서 final_generalization_gap이 0.04 이하, overfit_score가 0.12 이하로 내려가는 것이다. validation이 5.554-5.558 범위에 머무르고 overfit_score만 낮아지면 stride 축은 seed134 과적합 완화에 의미 있다고 본다. final_val_loss가 5.57 이상이면 overlapping window가 데이터 분포를 바꾸거나 noisy optimization을 만들어 저손실 장점을 훼손한 것으로 판단한다.
- 실제 결과: final_val_loss=5.557267347971599, gap=-0.016087373097737334, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0161, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 같은 stride=24 조건을 seed151 또는 seed202 저손실 경로에 반복해 평균 validation과 overfit_score가 동시에 개선되는지 확인한다. 세 seed 평균에서 low-loss 경로의 validation 장점을 유지하며 overfit_score가 낮아지면 stride=24를 기본 데이터 window 후보로 승격한다. / 과적합 시 gap이나 overfit_score가 줄지 않으면 seed134의 문제는 데이터 window 부족보다 high learning_rate의 train 편향에 가깝다고 본다. 그 경우 stride 축은 중단하고 run052 기준에서 max_steps=70 조기 종료 또는 seed별 하이브리드 전략을 다음 후보로 둔다. validation이 크게 악화되면 stride는 null로 되돌린다.

## run 057 - 2026-06-02T23:43:38+00:00

- 보고서: `docs/train/runs/run_057.md`
- 이번 가설: seed=151 저손실 경로에서 stride=24 재현성 검증: run056은 seed=134에서 stride를 null에서 24로 줄이자 final_val_loss=5.557267로 저손실 범위를 유지하면서 gap=-0.016087, overfit_score=0.0까지 낮춰 overfit-aware best가 되었다. run051은 seed=151, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 final_val_loss=5.553612로 pure validation은 뛰어났지만 gap=0.030143, overfit_score=0.086745로 아직 train 편향이 남아 있었다. 따라서 run051과 동일한 모델/함수/학습 조건에서 stride만 24로 줄이면, seed151에서도 validation을 크게 잃지 않으면서 gap과 overfit_score를 낮춰 stride=24가 seed134 특이 효과가 아니라 평균적으로 유효한 데이터 window 축인지 확인할 수 있다.
- 근거: 최근 결과는 learning_rate=0.0003 + max_steps=80 + gelu_exact + drop_rate=0.12가 pure validation loss 최저권을 만들고, learning_rate=0.000275는 과적합을 낮추는 대신 validation 비용을 만든다는 것을 보여줬다. run056은 learning_rate를 낮추지 않고 데이터 window만 바꿔 seed134의 과적합을 크게 줄였기 때문에, 이 축이 가장 유망한 다음 후보가 되었다. seed151은 seed134보다 과적합이 덜하지만 seed202보다 gap이 커서, stride 효과의 재현성과 과도 regularization 위험을 동시에 확인하기 좋은 중간 난이도 seed다. Transformer 구조, activation, attention, parameter_count는 모두 유지한다.
- 바꾼 변수: `{"stride": 24}`
- 기대 결과: 성공 기준은 run051 대비 final_generalization_gap과 overfit_score가 낮아지고 final_val_loss가 5.56 이하에 머무는 것이다. 특히 final_val_loss가 5.553-5.558 범위에 남으면서 overfit_score가 0.05 이하로 내려가면 stride=24를 저손실 계열의 강한 평균 후보로 본다. final_val_loss가 5.565 이상으로 악화되면 seed151에서는 overlapping window가 under-training 또는 데이터 분포 변화를 만든 것으로 판단한다.
- 실제 결과: final_val_loss=5.555843353271484, gap=-0.018288731575012207, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0183, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 seed202에서도 같은 stride=24 조건을 반복해 세 seed 평균을 완성한다. seed202에서도 validation을 유지하고 gap/overfit_score가 낮아지면 stride=24를 기본 데이터 window 후보로 승격하고, 이후 context_length=64 또는 max_steps 경계 실험을 보수적으로 진행한다. / 과적합 시 gap이나 overfit_score가 줄지 않으면 run056은 seed134 특이 개선일 수 있다. 그 경우 seed134에만 stride=24를 적용하는 하이브리드 전략을 문서화하고, seed151/202는 stride=null 저손실 경로를 유지한다. validation이 크게 악화되면 stride 축은 seed134 전용 완화책으로 제한한다.

## run 058 - 2026-06-02T23:49:25+00:00

- 보고서: `docs/train/runs/run_058.md`
- 이번 가설: seed=202 저손실 경로에서 stride=24 세 번째 seed 검증: run056(seed134)과 run057(seed151)은 learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 stride를 24로 줄였을 때 validation 손실을 저손실 범위에 유지하면서 final_generalization_gap을 음수로 만들고 overfit_score를 0.0까지 낮췄다. run050은 seed=202의 기존 stride=null 저손실 best로 final_val_loss=5.553959, gap=0.007347, overfit_score=0.041182를 보였다. 따라서 run050과 동일한 모델/함수/학습 조건에서 stride만 24로 줄이면, seed202에서도 validation을 크게 잃지 않으면서 gap과 overfit_score를 낮춰 stride=24를 세 seed 평균 기본 후보로 승격할 수 있는지 확인한다.
- 근거: 최근 두 번의 실험은 구조나 parameter_count를 바꾸지 않고 데이터 window만 촘촘히 만드는 stride=24가 과적합 지표를 강하게 줄인다는 신호를 줬다. seed134는 어려운 seed였고 seed151은 중간 난이도였는데 둘 다 overfit_score=0.0이 되었다. 이제 seed202는 기존 pure validation이 가장 좋은 seed라서, stride=24가 단순히 어려운 seed의 regularization 도구인지, 아니면 저손실 best seed에서도 과적합-aware score를 개선하는 범용 데이터 조건인지 판단할 수 있다. MPS balanced 장비에서 80 step은 계속 짧은 회차라 안전하다.
- 바꾼 변수: `{"stride": 24}`
- 기대 결과: 성공 기준은 run050 대비 final_generalization_gap과 overfit_score가 낮아지고 final_val_loss가 5.56 이하에 머무는 것이다. 특히 final_val_loss가 5.553-5.558 범위에 남으면서 overfit_score가 0.02 이하 또는 0.0으로 내려가면 stride=24를 세 seed 평균 기본 후보로 본다. final_val_loss가 5.565 이상이면 seed202에서는 overlap window가 best validation을 훼손한 것으로 판단한다.
- 실제 결과: final_val_loss=5.5566205978393555, gap=-0.006533384323120117, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0065, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 stride=24 + learning_rate=0.0003 + drop_rate=0.12 + gelu_exact를 기본 데이터 window 후보로 승격한다. 다음 실험은 같은 stride=24 위에서 context_length=64 또는 max_steps=90처럼 학습/문맥 확장 축을 보수적으로 한 번에 하나씩 확인한다. / 과적합 시 gap이나 overfit_score가 줄지 않으면 seed202는 stride=null의 run050을 pure validation 기준으로 유지하고, stride=24는 seed134/151에 더 강한 안정화 축으로 제한한다. validation이 크게 악화되면 stride=24의 세 seed 평균 이득을 다시 계산해 seed별 하이브리드 전략을 문서화한다.

## run 059 - 2026-06-02T23:54:39+00:00

- 보고서: `docs/train/runs/run_059.md`
- 이번 가설: stride=24 안정 후보 위에서 context_length=64 문맥 확장 검증: run056, run057, run058은 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 세 seed 모두 overfit_score=0.0을 만들었다. 그중 run057(seed151)은 final_val_loss=5.555843, gap=-0.018289로 현재 overfit-aware best다. 따라서 run057을 기준으로 Transformer 구조와 모델 크기는 유지하고 context_length를 48에서 64로 늘리며 stride도 24에서 32로 비례 조정하면, 50% overlap 데이터 window의 안정성은 유지하면서 더 긴 문맥이 validation loss를 더 낮출 수 있는지 확인한다.
- 근거: 세 seed 반복으로 stride=24는 seed 특이 효과가 아니라 안정적인 데이터 window 축임이 확인됐다. 이제 같은 안정 조건 위에서 다음으로 해석 가능한 작은 실험은 문맥 길이 확장이다. context_length=64는 모델 구조 순서를 바꾸지 않고 position embedding 길이와 학습 샘플 문맥만 늘린다. stride=32는 기존 context_length=48/stride=24와 같은 50% overlap 비율을 유지하므로, 단순히 overlap을 더 늘리는 실험이 아니라 더 긴 문맥 자체의 효과를 보는 데이터 window geometry 테스트가 된다. MPS balanced 장비에서 batch_size=8, max_steps=80은 여전히 짧은 회차로 안전하지만 tokens_per_sec 변화도 함께 확인한다.
- 바꾼 변수: `{"context_length": 64, "stride": 32}`
- 기대 결과: 성공 기준은 run057 대비 overfit_score가 0.02 이하 또는 0.0을 유지하고 final_val_loss가 5.556 이하에 머무르거나 개선되는 것이다. final_val_loss가 5.553-5.556 범위에 있고 gap이 안정적이면 context_length=64/stride=32를 stride 안정 후보 위의 개선 방향으로 본다. final_val_loss가 5.565 이상이면 더 긴 문맥이 작은 corpus와 80 step 조건에서 under-training 또는 데이터 희소성을 만든 것으로 판단한다.
- 실제 결과: final_val_loss=5.6005859375, gap=0.05377376079559326, overfit_score=0.14014971256256104, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0538, overfit_score=0.1401. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 성공하면 context_length=64/stride=32를 seed134 또는 seed202에 반복해 세 seed 평균을 확인한다. 세 seed에서도 overfit_score가 낮고 validation이 유지되면 이 window geometry를 기본 후보로 승격하고, 이후 max_steps=90을 한 번에 하나씩 확인한다. / 과적합 시 gap이나 overfit_score가 다시 커지면 context_length=64는 안정 후보가 아니라고 보고 context_length=48/stride=24로 되돌린다. validation이 크게 나빠지면 더 긴 문맥은 현 corpus와 80 step에서 under-training으로 판단하고, 다음에는 context_length를 유지하지 말고 max_steps=90만 단일축으로 확인한다.

## run 060 - 2026-06-02T23:59:37+00:00

- 보고서: `docs/train/runs/run_060.md`
- 이번 가설: context_length=64 실패 이후 best window로 복귀한 max_steps=90 단일축 검증: run059는 context_length=64, stride=32가 final_val_loss=5.600586, gap=0.053774, overfit_score=0.140150으로 악화되어 더 긴 문맥이 현재 corpus와 80-step 조건에서 안정적이지 않다는 신호를 줬다. 반면 run056, run057, run058의 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact 계열은 세 seed 모두 overfit_score=0.0을 유지했다. 따라서 현재 overfit-aware best인 run057(seed151) 설정으로 되돌리고 max_steps만 80에서 90으로 늘리면, 안정적인 overlapping window 위에서 추가 optimization이 validation을 더 낮추는지 또는 train-only 개선으로 gap을 다시 키우는지 분리해 확인할 수 있다.
- 근거: 최근 evidence는 stride=24가 구조 변경 없이 과적합 신호를 줄이는 가장 강한 축이고, context_length=64는 validation과 overfit 지표를 동시에 악화시키는 축임을 보여준다. 다음 실험은 새 함수나 모델 용량을 건드리지 않고 best 근방에서 학습 길이만 소폭 늘리는 실험이므로 해석이 쉽다. 성공하면 stride=24 조합이 더 긴 학습에서도 안정적이라는 근거가 되고, 실패하면 현재 조건의 early-stop 경계가 80 step 근처라는 결론을 얻는다. MPS balanced 하드웨어에서 90 step은 여전히 짧은 회차라 자동화 점유 시간이 안전하다.
- 바꾼 변수: `{"max_steps": 90}`
- 기대 결과: 성공 기준은 run057 대비 final_val_loss가 5.555 이하로 내려가거나 비슷하게 유지되면서 final_generalization_gap이 0.02 이하, overfit_score가 0.03 이하에 머무는 것이다. final_train_loss만 낮아지고 final_val_loss가 5.56 이상으로 오르거나 gap이 양수로 크게 커지면 90 step은 train-only fitting으로 판단한다. validation이 5.553대까지 내려가고 gap도 안정적이면 max_steps=90을 seed134 또는 seed202로 반복할 가치가 있다.
- 실제 결과: final_val_loss=5.551509380340576, gap=-0.009822964668273926, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0098, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 max_steps=90이 run057에서 validation을 개선하고 overfit_score를 낮게 유지하면 같은 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건으로 seed202 또는 seed134에 반복해 학습 길이 증가가 seed151 특이 효과인지 확인한다. 세 seed 평균에서도 안정적이면 max_steps=90을 stride=24 기본 후보로 승격한다. / 과적합 시 gap이나 overfit_score가 커지면 max_steps=90은 현재 작은 corpus에서 과도 학습으로 보고 max_steps=80을 기본값으로 유지한다. 그 경우 다음 실험은 max_steps를 더 늘리지 않고 activation_name=silu 또는 ffn_mult=3처럼 구조 순서를 바꾸지 않는 작은 함수/용량 축으로 이동하거나, seed 반복으로 run057의 안정성을 더 확인한다.

## run 061 - 2026-06-03T00:04:35+00:00

- 보고서: `docs/train/runs/run_061.md`
- 이번 가설: run060의 max_steps=90 개선이 seed151 특이 효과인지 확인하는 seed202 반복 검증: run060은 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact 조건에서 max_steps를 80에서 90으로 늘리자 final_val_loss=5.551509, gap=-0.009823, overfit_score=0.0으로 새 best가 되었다. 다만 run060은 seed151 한 번의 결과이므로, 같은 구조와 학습 조건에서 seed만 202로 바꾸면 max_steps=90이 안정적인 추가 optimization인지, 또는 seed151에서만 validation이 운 좋게 내려간 것인지 확인할 수 있다.
- 근거: 현재 가장 중요한 미해결 질문은 새 함수나 용량 축이 아니라 run060의 재현성이다. run058은 seed202, context_length=48, stride=24, max_steps=80에서 final_val_loss=5.556621, gap=-0.006533, overfit_score=0.0으로 이미 안정적이었다. 같은 seed202에 max_steps=90을 적용하면 run058 대비 학습 길이만 늘린 효과를 볼 수 있고, run060과 비교하면 seed 간 분산도 해석할 수 있다. context_length=64는 run059에서 명확히 실패했으므로 유지하지 않고, Transformer 구조와 함수 조합은 그대로 둔다. MPS balanced 하드웨어에서 90 step은 약 1초대라 자동화 주기에도 안전하다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 seed202의 기존 run058 대비 final_val_loss가 낮아지거나 5.556 이하를 유지하면서 final_generalization_gap이 0.02 이하, overfit_score가 0.03 이하에 머무는 것이다. final_val_loss가 5.553 이하로 내려가고 overfit_score가 0.0 근처이면 max_steps=90을 seed151/202 공통 개선 후보로 본다. final_val_loss가 5.565 이상으로 악화되거나 gap이 양수로 크게 커지면 90 step은 seed202에서는 train-only fitting 또는 불안정한 추가 학습으로 판단한다.
- 실제 결과: final_val_loss=5.544761975606282, gap=-0.005078832308451631, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0051, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed202에서도 max_steps=90이 validation을 개선하고 overfit_score를 낮게 유지하면 다음 실험은 seed134 stress test로 이동한다. seed134에서도 안정적이면 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact, max_steps=90을 기본 후보로 승격하고 이후 activation_name=silu 같은 함수 교체는 이 기준 위에서만 작은 단일축으로 확인한다. / 과적합 시 seed202에서 gap이나 overfit_score가 커지면 run060은 seed151 특이 개선일 수 있다. 그 경우 max_steps=90을 기본으로 승격하지 말고 seed151 전용 후보로 보류한다. 다음 실험은 max_steps=80으로 되돌린 뒤 seed 반복을 더 하거나, max_steps=85처럼 더 작은 학습 길이 경계를 확인한다.

## run 062 - 2026-06-03T00:09:33+00:00

- 보고서: `docs/train/runs/run_062.md`
- 이번 가설: max_steps=90 후보의 seed134 stress test: run060(seed151)과 run061(seed202)은 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact, max_steps=90 조건에서 각각 final_val_loss=5.551509와 5.544762를 기록했고 overfit_score=0.0을 유지했다. 하지만 seed134는 이전 run052에서 같은 저손실 계열이 gap=0.044721, overfit_score=0.139939까지 커졌던 가장 까다로운 seed다. 따라서 run061의 구조와 학습 조건을 유지하고 seed만 134로 바꾸면, max_steps=90과 stride=24 조합이 어려운 seed에서도 과적합 없이 validation을 개선하는지 확인할 수 있다.
- 근거: 현재 evidence는 세 단계로 정리된다. 첫째, stride=24는 run056-run058에서 세 seed 모두 overfit_score=0.0을 만들며 가장 강한 데이터 window 안정화 축이었다. 둘째, context_length=64는 run059에서 validation과 gap을 동시에 악화시켜 중단해야 할 축이다. 셋째, max_steps=90은 seed151과 seed202에서 validation을 더 낮추면서 overfit_score를 유지했다. 남은 핵심 검증은 seed134다. seed134는 high learning_rate 계열에서 train 편향이 커졌던 stress seed이므로, 여기서 성공하면 max_steps=90을 기본 후보로 승격할 근거가 된다. MPS balanced 환경에서 90 step은 약 1초대라 자동화 실행 단위로 안전하다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed134의 기존 stride=24/max_steps=80 run056 대비 final_val_loss가 5.557 이하로 유지되거나 개선되고, final_generalization_gap이 0.02 이하, overfit_score가 0.03 이하에 머무는 것이다. final_val_loss가 5.55대 초반으로 내려가면서 gap이 음수 또는 작은 양수이면 max_steps=90은 세 seed 공통 안정 후보가 된다. 반대로 final_train_loss만 낮아지고 final_val_loss가 5.565 이상이거나 gap이 0.04 이상으로 커지면 seed134에서는 90 step이 train-only fitting을 만든 것으로 판단한다.
- 실제 결과: final_val_loss=5.546862761179606, gap=-0.005851467450459502, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0059, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed134에서도 max_steps=90이 validation을 유지하거나 개선하고 overfit_score를 낮게 유지하면 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact, max_steps=90을 현재 기본 후보로 승격한다. 다음 실험은 새 capacity를 키우지 않고 activation_name=silu 또는 ffn_mult=3처럼 구조 순서를 유지하는 작은 함수/용량 축을 이 기준 위에서 단일축으로 테스트한다. / 과적합 시 seed134에서 gap이나 overfit_score가 커지면 max_steps=90은 seed151/202에는 좋지만 stress seed에는 과도한 학습으로 본다. 그 경우 max_steps=80을 seed134 기본값으로 유지하거나 max_steps=85 경계 실험을 진행한다. 또한 과적합이 강하면 drop_rate 증가 또는 learning_rate=0.000275 안정 경로로 되돌리는 보수적 실험을 우선한다.

## run 063 - 2026-06-03T00:14:46+00:00

- 보고서: `docs/train/runs/run_063.md`
- 이번 가설: 현재 기본 후보 위에서 activation_name=silu 단일축 함수 교체 검증: run060, run061, run062는 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, gelu_exact, max_steps=90 조건이 seed151/202/134 모두에서 overfit_score=0.0을 유지한다는 것을 보여줬다. 이제 구조나 parameter_count를 바꾸지 않는 activation 교체 축으로 넘어가도 된다. 따라서 현재 best인 run061(seed202)을 기준으로 activation_name만 gelu_exact에서 silu로 바꾸면, SiLU의 부드러운 gating-like 곡선이 안정적인 데이터 window와 90-step 학습 위에서 validation loss를 더 낮추는지 확인할 수 있다.
- 근거: max_steps=90 기준선은 세 seed에서 통과했고, context_length=64는 실패했으며, stride=24는 가장 강한 안정화 축으로 확인됐다. 다음 실험은 용량을 키우거나 Transformer 순서를 바꾸는 대신 FFN 내부 비선형 함수만 교체하는 것이 가장 해석 가능하다. silu는 swiglu/geglu처럼 gated FFN으로 parameter_count를 바꾸지 않고, mish보다 먼저 볼 만한 부드러운 activation 대안이다. seed202는 현재 best run061을 만든 seed라서 exploitation 관점에서 작은 개선 가능성을 확인하기 좋다. 성공하면 seed134 stress test로 일반화 여부를 확인하고, 실패하면 gelu_exact를 현재 기준 activation으로 유지한다.
- 바꾼 변수: `{"activation_name": "silu"}`
- 기대 결과: 성공 기준은 run061 대비 final_val_loss가 5.545 이하로 같거나 더 낮고, final_generalization_gap이 0.02 이하, overfit_score가 0.03 이하를 유지하는 것이다. final_val_loss가 5.54대 초반으로 내려가면 silu를 current baseline 위의 유망 activation 후보로 본다. final_val_loss가 5.555 이상으로 악화되거나 gap이 양수로 커지면 silu가 현 작은 corpus와 90-step 조건에서 gelu_exact보다 덜 맞는 것으로 판단한다.
- 실제 결과: final_val_loss=5.544584592183431, gap=-0.0048071543375654, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0048, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 silu가 seed202에서 validation을 개선하고 overfit_score를 낮게 유지하면 다음 실험은 같은 silu 설정을 seed134에 반복해 stress seed에서도 안정적인지 확인한다. seed134도 통과하면 seed151까지 반복해 silu를 gelu_exact와 세 seed 평균으로 비교한다. / 과적합 시 silu에서 gap이나 overfit_score가 커지거나 validation이 악화되면 gelu_exact를 현재 기준 activation으로 유지한다. 다음 실험은 activation을 더 밀기보다 ffn_mult=3처럼 parameter_count를 줄이는 용량 축 또는 mish 같은 다른 smooth activation을 별도 단일축으로 작게 확인한다.

## run 064 - 2026-06-03T00:20:43+00:00

- 보고서: `docs/train/runs/run_064.md`
- 이번 가설: silu activation이 seed202에서 gelu_exact 기준선을 아주 작게 앞섰고 overfit_score=0.0을 유지했다. seed134는 이전 고학습률 실험에서 과적합 신호가 강하게 드러났던 stress seed이므로, 같은 안정 조건에서 silu를 반복하면 이 개선이 우연한 seed 효과인지 실제 activation 후보인지 구분할 수 있다.
- 근거: 현재 best run063은 activation_name=silu, seed=202, final_val_loss=5.544585, final_generalization_gap=-0.004807, overfit_score=0.0으로 안정적이다. 바로 더 큰 구조나 길이로 확장하기보다 seed134에서 같은 activation을 검증하면 과적합 없이 validation 개선이 재현되는지 확인할 수 있다. 비교 기준은 run062의 gelu_exact seed134 결과(final_val_loss=5.546863, gap=-0.005851, overfit_score=0.0)다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 final_val_loss가 run062의 5.546863과 같거나 낮고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. 이 경우 silu는 seed202 단발 개선이 아니라 안정적인 후보로 볼 수 있다.
- 실제 결과: final_val_loss=5.546693166097005, gap=-0.0056075255076093455, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0056, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 silu를 seed151에서도 반복해 3-seed 평균을 gelu_exact의 3-seed 평균과 비교한다. 평균 validation이 개선되고 gap이 안정적이면 다음에는 ffn_mult=3 또는 dropout 위치를 한 축씩 검증한다. / 과적합 시 silu 후보를 보류하고 gelu_exact 안정 기준선으로 되돌린다. 이후 regularization을 늘리기보다 activation 계열(mish 또는 quick_gelu)을 같은 seed134 조건에서 단일 축으로 비교한다.

## run 065 - 2026-06-03T00:24:59+00:00

- 보고서: `docs/train/runs/run_065.md`
- 이번 가설: silu activation 후보의 3-seed 검증을 완성한다. run063(seed202)과 run064(seed134)는 activation_name=silu 조건에서 final_val_loss를 gelu_exact 대응 run보다 아주 작게 낮추면서 overfit_score=0.0을 유지했다. 남은 seed151에서도 같은 조건을 반복하면 silu 개선이 seed202/134에만 국한된 우연인지, 아니면 현재 안정 baseline 위에서 평균적으로 유효한 함수 교체인지 판단할 수 있다.
- 근거: 현재 안정 baseline은 context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, max_steps=90, tie_embeddings=True, attention_impl=sdpa, ffn_dropout_position=none이다. gelu_exact 기준은 seed151/202/134에서 각각 run060=5.551509, run061=5.544762, run062=5.546863으로 모두 overfit_score=0.0이었다. silu는 seed202에서 5.544585로 run061보다 미세하게 좋았고, seed134에서 5.546693으로 run062보다 미세하게 좋았다. 따라서 seed151에서 run060과 비교하면 silu의 평균 효과와 seed variance를 해석할 수 있다. 구조 순서, parameter_count, context/stride, optimizer 조건은 그대로 유지하므로 결과 해석은 activation 교체와 seed variance에 집중된다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 seed151의 gelu_exact 기준 run060(final_val_loss=5.551509, gap=-0.009823, overfit_score=0.0)과 같거나 더 낮은 final_val_loss를 기록하고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. 이 경우 silu는 3-seed 평균에서 gelu_exact를 소폭 앞서는 후보로 승격할 수 있다. final_val_loss가 5.555 이상이거나 gap이 양수로 커지면 silu의 장점은 seed202/134에 치우친 것으로 본다.
- 실제 결과: final_val_loss=5.551222006479899, gap=-0.009545048077900908, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0095, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 silu 3-seed 평균을 gelu_exact 3-seed 평균과 문서화해 비교한다. 평균 validation이 낮고 overfit_score 평균이 0.0에 머무르면 silu를 새 activation 기준 후보로 두고, 다음 실험은 parameter_count를 줄이는 ffn_mult=3 또는 dropout 위치를 after_activation으로 바꾸는 단일축 안정성 테스트로 이동한다. / 과적합 시 seed151에서 silu가 overfit_score를 키우거나 validation을 악화하면 gelu_exact를 기본 activation으로 유지한다. 그 다음에는 silu를 더 밀지 않고 quick_gelu 또는 mish를 seed202 단일축으로 비교하거나, ffn_mult=3으로 capacity를 줄여 안정성을 확인한다.

## run 066 - 2026-06-03T00:30:18+00:00

- 보고서: `docs/train/runs/run_066.md`
- 이번 가설: silu 3-seed 검증이 통과했으므로, 현재 안정 후보에서 FFN 내부 폭을 줄여도 validation과 과적합 안정성이 유지되는지 확인한다. ffn_mult를 4에서 3으로 낮추면 Transformer block 순서와 attention/activation 함수는 유지하면서 parameter_count만 줄어든다. 만약 seed202 best 조건에서 final_val_loss가 크게 악화되지 않고 overfit_score=0.0을 유지한다면 현재 모델은 FFN 폭 4가 약간 과한 용량일 수 있고, 더 작은 모델이 같은 일반화 성능을 낼 수 있다.
- 근거: run063, run064, run065는 activation_name=silu, context_length=48, stride=24, learning_rate=0.0003, drop_rate=0.12, max_steps=90 조건이 seed202/134/151에서 모두 overfit_score=0.0을 유지한다는 것을 보여줬다. silu 3-seed 평균은 gelu_exact보다 아주 작게 낮아 함수 후보로 유지할 근거가 생겼지만, 개선 폭은 작다. 이제 같은 함수 후보 위에서 capacity 축을 하나만 줄이면, 과적합 없이 validation이 유지되는지와 parameter_count/tokens_per_sec 개선 가능성을 동시에 확인할 수 있다. seed202는 현재 best run063을 만든 seed이므로 ffn_mult=3의 손실 비용을 가장 민감하게 볼 수 있다.
- 바꾼 변수: `{"ffn_mult": 3}`
- 기대 결과: 성공 기준은 run063(seed202, silu, ffn_mult=4)의 final_val_loss=5.544585 대비 final_val_loss가 5.548 이하에 머무르고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. parameter_count가 줄면서 validation 손실 증가가 0.003-0.004 이내라면 ffn_mult=3은 효율 후보가 된다. final_val_loss가 5.555 이상으로 올라가면 현재 corpus와 90-step 조건에서 FFN 폭 3은 underfit 또는 표현력 부족으로 본다.
- 실제 결과: final_val_loss=5.541161855061849, gap=-0.000325004259745576, overfit_score=0.013246536254882812, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0003, overfit_score=0.0132. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 ffn_mult=3이 seed202에서 validation을 유지하면 seed134 stress test로 반복한다. seed134에서도 overfit_score=0.0과 낮은 validation을 유지하면 ffn_mult=3을 silu baseline의 효율 후보로 두고 seed151까지 평균 비교를 완성한다. / 과적합 시 ffn_mult=3에서 validation이 악화되거나 gap이 커지면 ffn_mult=4를 유지한다. 그 다음에는 capacity를 더 줄이지 않고 ffn_dropout_position=after_activation 또는 activation_name=mish 같은 함수/드롭아웃 위치 단일축을 seed202에서 비교한다.

## run 067 - 2026-06-03T00:35:12+00:00

- 보고서: `docs/train/runs/run_067.md`
- 이번 가설: ffn_mult=3으로 줄인 silu 후보가 seed202에서 새 best를 만들었으므로, 같은 더 작은 FFN 폭이 seed134 stress seed에서도 validation과 과적합 안정성을 유지하는지 검증한다. seed134는 이전 고학습률 계열에서 과적합 신호가 강했던 seed라서, 여기서 ffn_mult=3이 통과하면 run066의 개선이 단순한 seed202 운이 아니라 작은 FFN이 현재 corpus에 더 잘 맞는다는 근거가 된다.
- 근거: run066은 seed202, activation_name=silu, ffn_mult=3에서 parameter_count를 478976에서 413184로 줄이면서 final_val_loss를 5.541162까지 낮췄고, final_generalization_gap=-0.000325, overfit_score=0.013247로 low risk를 유지했다. 비교 기준인 run063(seed202, silu, ffn_mult=4)은 final_val_loss=5.544585였다. 그러나 seed202는 현재 best를 자주 만든 seed이므로, 까다로운 seed134에서 같은 capacity 축소가 유지되는지 확인해야 한다. seed134의 직접 기준은 run064(seed134, silu, ffn_mult=4)의 final_val_loss=5.546693, gap=-0.005608, overfit_score=0.0이다. 이번 실험은 seed만 134로 바꾸고 ffn_mult=3 후보를 유지하므로, seed variance와 capacity 축소의 일반화 여부를 분리해 볼 수 있다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed134의 silu ffn_mult=4 기준 run064(final_val_loss=5.546693)과 같거나 낮은 final_val_loss를 기록하고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.548 이하이고 parameter_count=413184를 유지하면 ffn_mult=3은 효율 후보로 강해진다. final_val_loss가 5.555 이상으로 오르거나 overfit_score가 0.08 이상으로 커지면 run066은 seed202 특이 개선 또는 seed134에서는 표현력 부족으로 판단한다.
- 실제 결과: final_val_loss=5.548690954844157, gap=-0.011530836423237822, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0115, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed134에서도 ffn_mult=3이 통과하면 seed151에서 같은 설정을 반복해 3-seed 평균을 완성한다. 세 seed 평균이 silu ffn_mult=4보다 낮거나 비슷하고 overfit_score 평균이 low risk이면 ffn_mult=3을 새 기본 FFN 폭 후보로 승격한다. / 과적합 시 seed134에서 validation이 악화되거나 gap/overfit_score가 커지면 ffn_mult=3을 seed202 전용 후보로 보류하고 ffn_mult=4를 일반 기준으로 유지한다. 그 다음 실험은 ffn_dropout_position=after_activation 또는 activation_name=mish처럼 capacity를 더 줄이지 않는 단일축으로 이동한다.

## run 068 - 2026-06-03T00:40:13+00:00

- 보고서: `docs/train/runs/run_068.md`
- 이번 가설: silu + ffn_mult=3 효율 후보의 3-seed 검증을 완성한다. run066(seed202)은 ffn_mult=3이 새 best를 만들었고, run067(seed134)은 ffn_mult=4보다 약간 높은 validation loss를 보였지만 overfit_score=0.0과 더 작은 parameter_count를 유지했다. 남은 seed151에서 같은 설정을 반복하면 ffn_mult=3이 평균적으로 ffn_mult=4를 대체할 수 있는 효율 후보인지, 아니면 seed202에 강하게 치우친 축소 모델인지 판단할 수 있다.
- 근거: 현재 ffn_mult=3 결과는 seed202 final_val_loss=5.541162, seed134 final_val_loss=5.548691이다. 대응되는 silu ffn_mult=4 기준은 seed202 run063=5.544585, seed134 run064=5.546693, seed151 run065=5.551222이다. ffn_mult=3은 seed202에서는 크게 이겼고 seed134에서는 약 0.002 손실이 있지만 overfit_score=0.0을 유지했다. seed151은 중간 난이도 seed이며, 이 결과가 5.551 이하 또는 비슷한 수준이면 ffn_mult=3의 3-seed 평균이 ffn_mult=4보다 낮거나 매우 근접할 가능성이 있다. 이번 실험은 seed만 151로 바꾸고 구조 순서, activation, optimizer, context/stride, dropout 위치를 고정하므로 평균 비교에 필요한 마지막 조각이다.
- 바꾼 변수: `{"seed": 151}`
- 기대 결과: 성공 기준은 seed151의 silu ffn_mult=4 기준 run065(final_val_loss=5.551222, gap=-0.009545, overfit_score=0.0)과 같거나 크게 벗어나지 않는 final_val_loss를 기록하고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.552 이하이면 ffn_mult=3은 3-seed 평균에서 강한 효율 후보가 된다. final_val_loss가 5.557 이상이면 seed151에서는 FFN 폭 축소가 underfit 또는 표현력 부족을 만든 것으로 본다.
- 실제 결과: final_val_loss=5.542542775472005, gap=-0.01850787798563669, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0185, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed151에서도 ffn_mult=3이 통과하면 ffn_mult=3과 ffn_mult=4의 silu 3-seed 평균을 문서화하고, parameter_count와 tokens_per_sec까지 포함해 새 기본 후보를 정한다. 평균 validation이 낮거나 거의 같으면 ffn_mult=3을 기본 FFN 폭 후보로 승격한다. / 과적합 시 seed151에서 validation이 크게 악화되거나 overfit_score가 커지면 ffn_mult=3을 seed202 전용 효율 후보로 보류하고 ffn_mult=4를 일반 기준으로 유지한다. 다음에는 ffn_dropout_position=after_activation 또는 activation_name=mish를 ffn_mult=4 기준에서 단일축으로 확인한다.

## run 069 - 2026-06-03T00:46:26+00:00

- 보고서: `docs/train/runs/run_069.md`
- 이번 가설: 새 기준 후보인 silu + ffn_mult=3에서 FFN dropout 위치를 none에서 after_activation으로 바꾸면, Transformer 구조와 parameter_count를 유지하면서 activation 이후 hidden representation에만 약한 regularization을 줄 수 있다. run068은 seed151에서 현재 best를 만들었고 gap이 음수라 과적합은 없으므로, 이번 실험은 regularization 위치 변경이 validation 안정성을 더 높이는지 아니면 underfit을 만드는지 확인하는 단일축 테스트다.
- 근거: run066, run067, run068로 silu + ffn_mult=3의 3-seed 검증이 끝났고, ffn_mult=3 평균 validation은 ffn_mult=4 평균보다 낮으면서 parameter_count도 478976에서 413184로 줄었다. 최신 run068은 final_val_loss=5.542543, final_generalization_gap=-0.018508, overfit_score=0.0으로 현재 best다. 다만 gap이 이미 음수라 추가 regularization이 반드시 필요한 상황은 아니다. 따라서 seed151 best 조건을 그대로 유지하고 ffn_dropout_position만 after_activation으로 바꾸면, dropout 위치가 일반화에 주는 효과를 구조 변경 없이 직접 비교할 수 있다.
- 바꾼 변수: `{"ffn_dropout_position": "after_activation"}`
- 기대 결과: 성공 기준은 run068 대비 final_val_loss가 같거나 낮아지거나, 적어도 5.545 이내로 유지되면서 final_generalization_gap이 0.02 이하이고 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.552 이상으로 악화되면 after_activation dropout은 현재 작은 FFN 후보에서는 regularization 이득보다 underfit 비용이 큰 것으로 판단한다.
- 실제 결과: final_val_loss=5.542765935262044, gap=-0.01826862494150827, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0183, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 after_activation이 run068과 동등하거나 더 낮은 validation을 보이면 seed202 또는 seed134에서 같은 조건을 반복해 3-seed 평균으로 ffn_dropout_position=none과 비교한다. / 과적합 시 overfit_score가 커지거나 validation이 악화되면 ffn_dropout_position=none을 유지하고, 다음에는 activation_name=mish 또는 learning_rate의 작은 하향 조정을 ffn_mult=3 baseline 위에서 단일축으로 확인한다.

## run 070 - 2026-06-03T00:50:43+00:00

- 보고서: `docs/train/runs/run_070.md`
- 이번 가설: seed202의 silu + ffn_mult=3 기준선에서 FFN dropout 위치를 none에서 after_activation으로 바꾸면, run066의 매우 낮은 validation loss는 유지하면서 작은 overfit_score 신호를 더 낮출 수 있다. run069(seed151)는 after_activation이 best를 넘지는 못했지만 overfit_score=0.0과 거의 같은 validation을 유지했으므로, 이번에는 seed202 matched baseline에서 regularization 위치 효과를 재검증한다.
- 근거: 현재 best는 run068(seed151, ffn_dropout_position=none, final_val_loss=5.542543, overfit_score=0.0)이고, 최신 run069(seed151, after_activation)는 final_val_loss=5.542766으로 best보다 0.000223 높지만 gap=-0.018269, overfit_score=0.0을 유지했다. seed202 기준선 run066은 ffn_mult=3 none 조건에서 final_val_loss=5.541162로 순수 validation loss는 가장 낮지만 gap=-0.000325, overfit_score=0.013247로 아주 작은 overfit-aware penalty가 붙었다. 따라서 seed202에서 dropout 위치만 after_activation으로 바꾸면, validation 이득을 유지하면서 gap/overfit_score를 안정화하는지 판단할 수 있다.
- 바꾼 변수: `{"ffn_dropout_position": "after_activation", "seed": 202}`
- 기대 결과: 성공 기준은 seed202 matched baseline run066과 비교해 final_val_loss가 5.544 이하를 유지하고, final_generalization_gap이 0.01 이하이며, overfit_score가 run066의 0.013247보다 같거나 낮아지는 것이다. final_val_loss가 5.548 이상으로 악화되면 after_activation은 seed202에서는 regularization 이득보다 underfit 비용이 큰 것으로 판단한다.
- 실제 결과: final_val_loss=5.541762669881185, gap=0.0006219943364458302, overfit_score=0.015762527783711455, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0006, overfit_score=0.0158. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed202에서도 after_activation이 validation을 유지하고 overfit_score를 낮추면 seed134에서 같은 조건을 반복해 ffn_dropout_position=after_activation의 3-seed 평균을 none과 비교한다. / 과적합 시 after_activation이 seed202에서 validation을 크게 악화시키거나 overfit_score를 높이면 dropout 위치 후보를 보류하고 ffn_dropout_position=none을 기본으로 유지한다. 다음에는 activation_name=mish 또는 learning_rate=0.000275를 ffn_mult=3 baseline 위에서 단일축으로 확인한다.

## run 071 - 2026-06-03T00:55:44+00:00

- 보고서: `docs/train/runs/run_071.md`
- 이번 가설: seed202의 silu + ffn_mult=3 기준선(run066)에서 learning_rate를 0.0003에서 0.000275로 아주 작게 낮추면, 낮은 validation loss는 유지하면서 train-only 개선 폭과 overfit_score를 줄일 수 있다. after_activation dropout은 run069와 run070에서 best를 넘지 못했고 seed202에서는 overfit_score가 오히려 커졌으므로, 이번에는 dropout 위치가 아니라 optimizer step size를 단일축으로 조정한다.
- 근거: run066은 seed202, silu, ffn_mult=3, ffn_dropout_position=none에서 final_val_loss=5.541162로 raw validation이 가장 낮았지만 final_generalization_gap=-0.000325, overfit_score=0.013247로 아주 작은 overfit-aware penalty가 있었다. run070은 같은 seed202에서 ffn_dropout_position=after_activation을 시도했지만 final_val_loss=5.541763으로 악화되고 gap=0.000622, overfit_score=0.015763으로 penalty도 커졌다. 따라서 dropout 위치 후보는 보류하고, run066 설정을 유지한 채 learning_rate만 0.000275로 낮추면 최종 train loss가 과도하게 앞서는 현상을 줄이면서 validation을 안정화할 수 있는지 확인할 수 있다.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: 성공 기준은 run066 대비 final_val_loss가 5.543 이하로 유지되고, final_generalization_gap이 0.0 이하 또는 거의 0에 머물며, overfit_score가 0.013247보다 낮아지는 것이다. final_val_loss가 5.546 이상으로 올라가면 learning_rate 하향이 optimization 부족을 만든 것으로 판단한다.
- 실제 결과: final_val_loss=5.546916961669922, gap=-0.003379344940185547, overfit_score=0.007137854894002871, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0034, overfit_score=0.0071. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 learning_rate=0.000275가 seed202에서 overfit_score를 줄이고 validation을 유지하면 seed151 또는 seed134에서 같은 learning_rate를 반복해 현재 best run068의 안정성을 평균으로 검증한다. / 과적합 시 learning_rate 하향에도 gap이나 overfit_score가 커지면 optimizer 조정보다는 seed202 자체의 저손실/근접 gap 특성으로 보고, ffn_dropout_position=none과 learning_rate=0.0003을 기본으로 유지한다. 다음에는 activation_name=mish를 ffn_mult=3 기준에서 단일축으로 확인한다.

## run 072 - 2026-06-03T01:00:46+00:00

- 보고서: `docs/train/runs/run_072.md`
- 이번 가설: 현재 overfit-aware best인 run068의 silu + ffn_mult=3 기준에서 activation_name만 mish로 바꾸면, 구조 순서와 parameter_count를 유지하면서 SiLU보다 더 부드러운 non-monotonic activation이 validation loss를 낮추거나 generalization gap 안정성을 유지할 수 있다. 최근 after_activation dropout과 learning_rate 하향은 best를 넘지 못했으므로, 이번에는 regularization/optimizer가 아니라 FFN 함수 교체 축으로 돌아간다.
- 근거: run068은 seed151, activation_name=silu, ffn_mult=3, ffn_dropout_position=none, learning_rate=0.0003에서 final_val_loss=5.542543, final_generalization_gap=-0.018508, overfit_score=0.0으로 현재 best다. run069는 같은 seed151에서 dropout 위치만 after_activation으로 바꿨지만 final_val_loss=5.542766으로 미세하게 악화됐다. run070은 seed202에서 after_activation이 overfit_score를 더 키웠고, run071은 learning_rate=0.000275가 overfit_score를 낮추는 대신 validation을 5.546917까지 악화시켰다. 따라서 현재 evidence는 best 설정의 optimizer/dropout을 건드리기보다, parameter_count와 Transformer 순서를 유지한 채 activation 함수만 비교하는 것이 더 해석 가능하다고 말한다.
- 바꾼 변수: `{"activation_name": "mish"}`
- 기대 결과: 성공 기준은 run068 대비 final_val_loss가 5.543 이하로 유지되거나 더 낮아지고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.55 이상으로 올라가면 mish는 현재 작은 corpus와 90-step 조건에서 optimization 비용 또는 activation mismatch가 큰 것으로 판단한다.
- 실제 결과: final_val_loss=5.542157967885335, gap=-0.017934600512186982, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0179, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 mish가 seed151에서 run068과 동등하거나 더 좋은 validation을 보이면 seed202에서 같은 설정을 반복해 silu ffn_mult=3 기준 run066과 비교한다. 두 seed가 통과하면 seed134 stress test로 3-seed 평균을 완성한다. / 과적합 시 mish에서 validation이 악화되거나 overfit_score가 커지면 activation_name=silu를 기본으로 유지한다. 다음에는 quick_gelu 또는 gelu_exact 재확인을 ffn_mult=3 기준에서 단일축으로 진행해 activation 계열의 순위를 좁힌다.

## run 073 - 2026-06-03T01:05:24+00:00

- 보고서: `docs/train/runs/run_073.md`
- 이번 가설: run072에서 새 best가 된 mish + ffn_mult=3 후보를 seed202에서 반복하면, mish 개선이 seed151에만 국한된 우연인지 아니면 silu를 대체할 수 있는 activation 후보인지 판단할 수 있다. 구조 순서, FFN 폭, optimizer, dropout 위치, context/stride는 그대로 두고 seed만 바꾸어 재현성을 확인한다.
- 근거: run072는 seed151에서 activation_name을 silu에서 mish로 바꾸자 final_val_loss가 run068의 5.542543에서 5.542158로 낮아지고 overfit_score=0.0을 유지해 새 best가 되었다. 그러나 seed151은 run068에서도 강했던 seed이므로, mish가 실제 activation 개선인지 확인하려면 seed202 반복이 필요하다. seed202의 직접 비교 기준은 run066(silu, ffn_mult=3, final_val_loss=5.541162, gap=-0.000325, overfit_score=0.013247)이며, mish가 이와 비슷한 validation을 유지하면서 gap/overfit_score를 낮추면 activation 후보로 강해진다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 final_val_loss가 seed202 silu 기준 run066의 5.541162와 크게 멀어지지 않는 5.544 이하에 머물고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. 특히 overfit_score가 run066의 0.013247보다 낮아지면 mish는 seed202에서도 안정성 이득이 있다고 본다. final_val_loss가 5.548 이상이면 mish 개선은 seed151 특이 효과일 가능성이 크다.
- 실제 결과: final_val_loss=5.5411020914713545, gap=0.0003427664438886424, overfit_score=0.015279928843180635, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0003, overfit_score=0.0153. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed202에서도 mish가 validation과 overfit_score를 유지하면 seed134 stress test를 진행해 mish + ffn_mult=3의 3-seed 평균을 silu + ffn_mult=3 평균과 비교한다. / 과적합 시 seed202에서 mish가 validation을 크게 악화시키거나 overfit_score를 키우면 mish를 seed151 후보로만 보류하고 activation_name=silu를 기본으로 유지한다. 다음에는 quick_gelu 또는 gelu_exact를 ffn_mult=3 기준에서 단일축으로 확인한다.

## run 074 - 2026-06-03T01:10:40+00:00

- 보고서: `docs/train/runs/run_074.md`
- 이번 가설: mish + ffn_mult=3 후보를 seed134 stress seed에서 반복하면, mish가 seed151/202에서만 좋은 우연인지 아니면 세 seed 평균에서도 silu를 대체할 수 있는 activation 후보인지 판단할 수 있다. 구조 순서, FFN 폭, optimizer, dropout 위치, context/stride는 그대로 유지하고 seed만 134로 바꿔 3-seed 검증을 완성한다.
- 근거: run072(seed151)는 mish가 silu 기준 run068보다 final_val_loss를 5.542543에서 5.542158로 낮추고 overfit_score=0.0을 유지해 새 best가 되었다. run073(seed202)는 mish가 final_val_loss=5.541102로 silu 기준 run066의 5.541162와 거의 같거나 미세하게 낮았지만 overfit_score=0.01528로 작은 penalty가 남았다. 이제 seed134를 확인해야 평균 판단이 가능하다. seed134에서 silu + ffn_mult=3 기준 run067은 final_val_loss=5.548691, gap=-0.011531, overfit_score=0.0이었고, silu + ffn_mult=4 기준 run064는 final_val_loss=5.546693이었다. mish가 이 범위를 유지하거나 개선하면 activation 후보로 강해진다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed134 silu ffn_mult=3 기준 run067의 final_val_loss=5.548691과 같거나 낮고, final_generalization_gap이 0.02 이하이며, overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.5467 이하로 내려가면 mish는 seed134에서도 ffn_mult=4 silu 기준까지 이기는 강한 후보가 된다. final_val_loss가 5.555 이상이면 mish의 개선은 seed151/202에 치우친 것으로 판단한다.
- 실제 결과: final_val_loss=5.548671881357829, gap=-0.011241157849629424, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0112, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 seed134에서도 mish가 통과하면 mish + ffn_mult=3의 3-seed 평균을 silu + ffn_mult=3, silu + ffn_mult=4 평균과 비교하고, 다음에는 best 계열에서 quick_gelu 또는 squared_relu 같은 activation ablation을 하나씩 확인한다. / 과적합 시 seed134에서 mish가 validation을 크게 악화시키거나 overfit_score를 키우면 mish를 seed151/202 유망 후보로 보류하고 activation_name=silu를 기본값으로 유지한다. 다음에는 quick_gelu 또는 gelu_exact를 ffn_mult=3 기준에서 단일축으로 확인한다.

## run 075 - 2026-06-03T01:20:54+00:00

- 보고서: `docs/train/runs/run_075.md`
- 이번 가설: mish + ffn_mult=3과 silu + ffn_mult=3이 세 seed에서 거의 같은 low-loss/low-overfit 영역을 형성했으므로, 같은 작은 FFN 후보에서 activation_name만 quick_gelu로 바꾸면 GELU 근사의 단순한 곡선이 mish/silu와 동등한 validation loss를 유지하면서 더 빠르거나 더 안정적인 대안이 될 수 있다. 우선 현재 best인 run072와 같은 seed151 조건에서 matched activation ablation을 시작한다.
- 근거: run072(mish, seed151)는 final_val_loss=5.542158, gap=-0.017935, overfit_score=0.0으로 현재 best다. 같은 seed151의 silu 기준 run068도 final_val_loss=5.542543, gap=-0.018508, overfit_score=0.0으로 매우 근접했고, mish의 seed134/202 반복(run073/run074)도 silu와 거의 같은 평균 영역에 머물렀다. 따라서 다음 연구 축은 regularization이나 capacity를 다시 건드리기보다, ffn_mult=3, context_length=48, stride=24, max_steps=90의 안정 조건 위에서 남은 activation 후보를 순차 비교하는 것이다. quick_gelu는 과거 짧은 40-step 계열에서 유망했지만 seed/context/stride 조건이 달라 현재 best 계열에서는 아직 직접 비교되지 않았다.
- 바꾼 변수: `{"activation_name": "quick_gelu"}`
- 기대 결과: 성공 기준은 seed151 matched baseline run072의 final_val_loss=5.542158과 run068의 5.542543에 근접한 5.543 이하를 유지하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. tokens_per_sec가 mish보다 높으면 quick_gelu는 속도까지 포함한 후보가 된다. final_val_loss가 5.548 이상이면 quick_gelu는 현재 90-step 작은 FFN 후보에서 mish/silu보다 뒤처지는 것으로 판단한다.
- 실제 결과: final_val_loss=5.542804876963298, gap=-0.018744428952534697, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0187, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 quick_gelu가 seed151에서 mish/silu와 동등하거나 더 좋으면 seed202에서 같은 설정을 반복해 run066/run073과 matched 비교한다. 두 seed가 통과하면 seed134 stress test까지 실행해 quick_gelu의 3-seed 평균과 처리량을 mish/silu 평균과 비교한다. / 과적합 시 quick_gelu에서 gap이나 overfit_score가 커지면 quick_gelu를 이전 40-step 계열 전용 후보로 보류하고, 현재 best 계열에서는 mish/silu를 유지한다. 다음에는 squared_relu 또는 gelu_exact를 activation 단일축으로 확인해 함수 계열 순위를 좁힌다.

## run 076 - 2026-06-03T01:25:48+00:00

- 보고서: `docs/train/runs/run_076.md`
- 이번 가설: run075에서 quick_gelu + ffn_mult=3은 seed151에서 mish/silu best 계열과 거의 같은 validation loss를 보이면서 overfit_score=0.0과 더 높은 tokens_per_sec를 기록했다. 같은 설정을 seed202에서 반복하면 quick_gelu가 단일 seed의 near-miss인지, 아니면 mish/silu와 3-seed 평균 비교에 올릴 만한 activation 후보인지 판단할 수 있다.
- 근거: 최신 run075는 final_val_loss=5.542805, final_generalization_gap=-0.018744, overfit_score=0.0, tokens_per_sec=30772.45로 안정적이었다. seed151 matched 기준인 run072(mish)는 final_val_loss=5.542158, run068(silu)는 5.542543이라 quick_gelu가 아주 근소하게 뒤졌지만 gap은 더 음수이고 속도도 mish보다 높았다. seed202는 같은 작은 FFN 후보에서 raw validation이 가장 낮게 나오는 동시에 작은 overfit penalty가 생기는 seed다. seed202 matched 비교 기준은 run066(silu, val=5.541162, overfit_score=0.013247)과 run073(mish, val=5.541102, overfit_score=0.015280)이므로, quick_gelu가 이 범위에 머물면 activation 후보로 계속 검증할 가치가 있다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 seed202 matched baselines인 run066/run073과 크게 멀어지지 않는 final_val_loss 5.544 이하, final_generalization_gap 0.02 이하, overfit_score 0.03 이하를 유지하는 것이다. tokens_per_sec가 mish seed202보다 높으면 quick_gelu는 속도 포함 후보로 남긴다. final_val_loss가 5.548 이상이거나 overfit_score가 0.05 이상이면 quick_gelu는 seed151에서는 안정적이지만 seed202 일반화가 부족한 것으로 본다.
- 실제 결과: final_val_loss=5.541934013366699, gap=-0.0004949569702148438, overfit_score=0.012966553370157285, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0005, overfit_score=0.0130. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 quick_gelu가 seed202에서도 통과하면 seed134 stress seed로 반복해 quick_gelu + ffn_mult=3의 3-seed 평균을 완성한다. 세 seed 평균에서 mish/silu와 거의 같고 처리량이 높으면 quick_gelu를 효율 activation 후보로 보류한다. / 과적합 시 seed202에서 quick_gelu의 gap이나 overfit_score가 커지면 quick_gelu를 현재 best 계열의 기본 후보에서 제외하고 mish/silu를 유지한다. 다음에는 squared_relu 또는 gelu_exact를 ffn_mult=3 기준에서 activation 단일축으로 확인한다.

## run 077 - 2026-06-03T01:30:36+00:00

- 보고서: `docs/train/runs/run_077.md`
- 이번 가설: quick_gelu + ffn_mult=3이 seed151과 seed202에서 모두 low-risk generalizing으로 통과했으므로, 같은 설정을 seed134 stress seed에서 반복하면 quick_gelu가 mish/silu와 3-seed 평균 비교에 올릴 수 있는 효율 activation 후보인지 확정할 수 있다.
- 근거: run075(seed151)는 final_val_loss=5.542805, gap=-0.018744, overfit_score=0.0으로 mish run072와 silu run068에 매우 근접했고, run076(seed202)는 final_val_loss=5.541934, gap=-0.000495, overfit_score=0.012967로 seed202 matched baselines인 silu run066(val=5.541162, overfit_score=0.013247)과 mish run073(val=5.541102, overfit_score=0.015280)의 같은 저손실 영역에 들어왔다. 남은 seed134는 이전 여러 실험에서 validation과 overfit 신호가 흔들렸던 stress seed이며, mish run074와 silu run067이 모두 약 5.54867대에 머물렀다. 따라서 seed134에서 quick_gelu가 이 범위를 유지하면 quick_gelu의 3-seed 평균과 처리량을 activation 후보 비교에 포함할 수 있다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed134 matched baselines인 silu run067(final_val_loss=5.548691, overfit_score=0.0)과 mish run074(final_val_loss=5.548672, overfit_score=0.0)에 근접하거나 더 낮은 final_val_loss 5.549 이하를 유지하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.5467 이하이면 seed134에서도 ffn_mult=4 silu 기준까지 이기는 강한 후보가 된다. 5.555 이상이거나 overfit_score가 0.05 이상이면 quick_gelu는 seed134 stress 조건에서 약한 것으로 본다.
- 실제 결과: final_val_loss=5.548875331878662, gap=-0.012146592140197754, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0121, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 quick_gelu가 seed134에서도 통과하면 quick_gelu, mish, silu의 ffn_mult=3 3-seed 평균 final_val_loss, overfit_score, tokens_per_sec를 비교한다. 평균 validation이 동등하고 처리량이 높으면 quick_gelu를 효율 activation 후보로 기록하고, 다음에는 squared_relu 또는 gelu_exact를 같은 기준에서 단일축으로 확인한다. / 과적합 시 quick_gelu가 seed134에서 validation을 크게 잃거나 overfit_score를 키우면 quick_gelu를 seed151/202 near-peer 후보로만 보류하고 mish/silu를 기본 activation 후보로 유지한다. 다음에는 squared_relu 또는 gelu_exact를 ffn_mult=3 기준에서 activation 단일축으로 확인한다.

## run 078 - 2026-06-03T01:35:58+00:00

- 보고서: `docs/train/runs/run_078.md`
- 이번 가설: mish, silu, quick_gelu는 ffn_mult=3 안정 세팅에서 모두 low-risk이지만 세 seed 평균 차이가 매우 작고 현재 best를 크게 넘지는 못했다. 같은 구조와 학습 조건에서 activation_name만 squared_relu로 바꾸면 ReLU 계열의 단순한 sparse positive activation과 제곱 비선형성이 작은 FFN 폭에서 더 강한 feature selectivity를 만들어 validation loss를 낮출 수 있는지 확인할 수 있다.
- 근거: 최근 run072-077은 context_length=48, stride=24, max_steps=90, ffn_mult=3, drop_rate=0.12 조건에서 activation 선택만 남은 안정 plateau를 보여준다. mish 3-seed 평균은 약 5.54398, silu 3-seed 평균은 약 5.54413, quick_gelu 3-seed 평균은 약 5.54454로 모두 비슷하며, best는 run072(mish, seed151, val=5.542158, overfit_score=0.0)다. quick_gelu는 세 seed 모두 통과했지만 평균상 mish/silu를 넘지 못했으므로, 다음에는 GELU/SiLU/Mish의 부드러운 activation 계열을 벗어나 squared_relu를 seed151 matched baseline에서 먼저 확인하는 것이 정보량이 크다. seed151은 run068/run072/run075에서 activation 간 차이를 민감하게 보여준 matched comparison seed다.
- 바꾼 변수: `{"activation_name": "squared_relu"}`
- 기대 결과: 성공 기준은 seed151 matched baselines인 run072(mish, final_val_loss=5.542158), run068(silu, 5.542543), run075(quick_gelu, 5.542805)에 근접한 final_val_loss 5.543 이하를 유지하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.548 이상이면 squared_relu는 현재 작은 FFN 후보에서 표현력 또는 optimization mismatch가 큰 것으로 본다. overfit_score가 커지면 ReLU 계열의 sharper activation이 train 쪽으로 치우친 신호로 해석한다.
- 실제 결과: final_val_loss=5.545124371846517, gap=-0.020150581995646455, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0202, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 squared_relu가 seed151에서 5.543 이하와 low-risk를 유지하면 seed202에서 반복해 run066/run073/run076의 저손실 seed와 비교한다. 두 seed가 통과하면 seed134 stress test까지 수행해 squared_relu의 3-seed 평균을 mish/silu/quick_gelu와 비교한다. / 과적합 시 squared_relu가 validation을 크게 잃거나 overfit_score를 키우면 ReLU 계열은 현재 안정 세팅에서 보류하고, 다음에는 gelu_exact + ffn_mult=3을 seed151에서 확인해 GELU 기준선이 작은 FFN 폭에서 얼마나 따라오는지 측정한다.

## run 079 - 2026-06-03T01:45:28+00:00

- 보고서: `docs/train/runs/run_079.md`
- 이번 가설: squared_relu는 seed151 matched ablation에서 overfit_score=0.0과 높은 처리량을 보였지만 final_val_loss=5.545124로 mish/silu/quick_gelu seed151 기준선보다 뒤처졌다. 같은 ffn_mult=3 안정 세팅에서 activation_name만 gelu_exact로 되돌리면, 작은 FFN 폭에서도 원래 GELU 계열 기준선이 smooth activation plateau와 얼마나 근접한지 확인할 수 있다.
- 근거: 최근 run072-078은 context_length=48, stride=24, max_steps=90, ffn_mult=3, drop_rate=0.12, learning_rate=0.0003 조건에서 모두 generalizing/low-risk를 유지했다. seed151 matched 비교에서 mish(run072)는 final_val_loss=5.542158, silu(run068)는 5.542543, quick_gelu(run075)는 5.542805였고, squared_relu(run078)는 5.545124로 더 나빴다. 따라서 ReLU 계열을 계속 반복하기보다, ffn_mult=4 기준에서 안정적이었던 gelu_exact를 ffn_mult=3에서도 직접 재측정해 activation 순위를 닫는 것이 다음 정보량이 크다.
- 바꾼 변수: `{"activation_name": "gelu_exact"}`
- 기대 결과: 성공 기준은 seed151 matched activation 기준선인 run072/run068/run075와 같은 5.543 이하 final_val_loss를 기록하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.545 이상이면 gelu_exact는 작은 FFN 폭에서 mish/silu/quick_gelu보다 뒤처지는 것으로 본다. overfit_score가 커지면 GELU 계열도 seed151에서 train 쪽으로 치우친 신호로 해석한다.
- 실제 결과: final_val_loss=5.542699495951335, gap=-0.018719395001729033, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0187, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 gelu_exact가 seed151에서 5.543 이하와 low-risk를 유지하면 seed202에서 반복해 run066/run073/run076의 저손실 seed와 비교한다. 두 seed가 통과하면 seed134 stress test까지 수행해 gelu_exact, mish, silu, quick_gelu의 ffn_mult=3 3-seed 평균을 정리한다. / 과적합 시 gelu_exact가 validation을 잃거나 overfit_score를 키우면 activation 탐색을 mish/silu/quick_gelu 후보로 좁히고, 다음에는 activation이 아니라 seed202의 작은 positive gap을 낮추는 가벼운 regularization 또는 weight_decay 단일축을 검토한다.

## run 080 - 2026-06-03T01:47:21+00:00

- 보고서: `docs/train/runs/run_080.md`
- 이번 가설: gelu_exact + ffn_mult=3은 seed151에서 final_val_loss=5.542699, final_generalization_gap=-0.018719, overfit_score=0.0으로 mish/silu/quick_gelu plateau에 합류했다. 같은 설정을 seed202에서 반복하면 gelu_exact가 작은 FFN 폭에서도 activation 후보로 재진입할 수 있는지, 아니면 seed151에서만 근접한 기준선인지 판단할 수 있다.
- 근거: 최근 run072-079는 context_length=48, stride=24, max_steps=90, ffn_mult=3, drop_rate=0.12, learning_rate=0.0003 조건에서 모두 generalizing/low-risk를 유지한다. run079는 seed151에서 gelu_exact가 run072(mish, 5.542158), run068(silu, 5.542543), run075(quick_gelu, 5.542805)에 매우 근접했고 squared_relu(run078, 5.545124)보다 명확히 좋았다. seed202는 이 세 activation의 저손실 비교가 이미 있는 축이다: run066(silu) val=5.541162, overfit_score=0.013247; run073(mish) val=5.541102, overfit_score=0.015280; run076(quick_gelu) val=5.541934, overfit_score=0.012967. 따라서 seed만 202로 바꾸면 gelu_exact의 seed variance와 activation 순위를 가장 직접적으로 측정할 수 있다.
- 바꾼 변수: `{"seed": 202}`
- 기대 결과: 성공 기준은 seed202 matched baselines인 run066/run073/run076과 같은 저손실 영역인 final_val_loss 5.544 이하를 유지하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.5415 이하이면 gelu_exact는 seed202에서도 mish/silu와 강하게 경쟁한다. final_val_loss가 5.548 이상이면 seed151 근접성은 seed 특이 효과로 본다.
- 실제 결과: final_val_loss=5.541786829630534, gap=-0.0004213651021318654, overfit_score=0.013099829355876835, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0004, overfit_score=0.0131. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 gelu_exact가 seed202에서도 5.544 이하와 low-risk를 유지하면 seed134 stress seed에서 반복해 gelu_exact의 3-seed 평균을 완성한다. 세 seed 평균이 mish/silu/quick_gelu와 동등하면 activation 선택은 품질보다 처리량과 overfit_score 평균 기준으로 정리한다. / 과적합 시 gelu_exact가 seed202에서 positive gap이나 overfit_score를 크게 키우면 gelu_exact를 seed151 near-peer 후보로만 보류하고, 기본 activation 후보는 mish/silu/quick_gelu로 유지한다. 다음에는 activation 탐색을 닫고 seed202의 작은 positive gap을 낮추는 weight_decay 또는 dropout 위치 단일축으로 이동한다.

## run 081 - 2026-06-03T01:52:17+00:00

- 보고서: `docs/train/runs/run_081.md`
- 이번 가설: gelu_exact + ffn_mult=3은 seed151과 seed202에서 모두 low-risk로 통과했으므로, 같은 설정을 seed134 stress seed에서 반복하면 gelu_exact가 mish/silu/quick_gelu와 3-seed 평균 비교에 들어갈 수 있는 activation 후보인지 확정할 수 있다.
- 근거: 최근 run079(seed151)는 gelu_exact + ffn_mult=3에서 final_val_loss=5.542699, gap=-0.018719, overfit_score=0.0을 기록했고, run080(seed202)는 final_val_loss=5.541787, gap=-0.000421, overfit_score=0.013100으로 seed202 matched activation baselines와 같은 저손실 영역에 들어왔다. 남은 seed134는 현재 작은 FFN 후보에서 activation 후보의 stress 비교를 완성하는 축이다. seed134 matched baselines는 silu run067 val=5.548691, mish run074 val=5.548672, quick_gelu run077 val=5.548875로 매우 좁게 모여 있다. 따라서 seed만 134로 바꾸면 gelu_exact의 평균 품질과 stress-seed 안정성을 직접 판단할 수 있다.
- 바꾼 변수: `{"seed": 134}`
- 기대 결과: 성공 기준은 seed134 matched baselines인 run067/run074/run077과 같은 final_val_loss 5.549 이하를 유지하고, final_generalization_gap이 0.02 이하이며 overfit_score가 0.03 이하로 유지되는 것이다. final_val_loss가 5.5469 이하이면 gelu_exact는 seed134에서 ffn_mult=4 gelu_exact run062까지 이기는 강한 후보가 된다. final_val_loss가 5.555 이상이거나 overfit_score가 0.05 이상이면 gelu_exact는 stress seed에서 약한 것으로 본다.
- 실제 결과: final_val_loss=5.548844178517659, gap=-0.012049476305644014, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0120, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 gelu_exact가 seed134에서도 5.549 이하와 low-risk를 유지하면 gelu_exact, mish, silu, quick_gelu의 ffn_mult=3 3-seed 평균 final_val_loss, overfit_score, tokens_per_sec를 정리한다. 평균 품질이 동등하면 처리량과 overfit_score 평균을 기준으로 기본 activation을 고르고, 이후에는 activation 탐색을 닫고 regularization 또는 optimization 단일축으로 이동한다. / 과적합 시 gelu_exact가 seed134에서 validation을 크게 잃거나 overfit_score를 키우면 gelu_exact는 seed151/202 near-peer 후보로 보류하고 기본 activation 후보를 mish/silu/quick_gelu로 좁힌다. 다음에는 activation 탐색보다 seed202의 작은 positive gap을 낮추는 weight_decay 또는 dropout 위치 단일축을 검토한다.

## run 082 - 2026-06-03T01:59:01+00:00

- 보고서: `docs/train/runs/run_082.md`
- 이번 가설: activation sweep 결과 mish가 3-seed 평균 final_val_loss에서 가장 앞서고 current best도 run072(mish, seed151)이므로, activation 탐색은 닫고 mish 후보의 seed202 작은 overfit penalty를 줄이는 optimization 단일축을 확인한다. run073의 mish + seed202 + max_steps=90에서 max_steps만 85로 줄이면 낮은 validation loss를 대부분 유지하면서 train 쪽 과진행과 overfit_score를 낮출 수 있다.
- 근거: ffn_mult=3, context_length=48, stride=24, drop_rate=0.12 조건에서 3-seed activation 평균은 mish mean_val=5.543977, silu mean_val=5.544132, gelu_exact mean_val=5.544444, quick_gelu mean_val=5.544538로 매우 좁지만 mish가 근소하게 가장 좋다. seed202 matched runs는 모두 낮은 raw validation과 작은 penalty를 보인다: mish run073 val=5.541102, gap=0.000343, overfit_score=0.015280; silu run066 val=5.541162, overfit_score=0.013247; quick_gelu run076 val=5.541934, overfit_score=0.012967; gelu_exact run080 val=5.541787, overfit_score=0.013100. 이전 run070의 after_activation dropout은 seed202에서 validation과 overfit_score를 모두 악화했고, run071의 learning_rate 하향은 overfit_score를 낮췄지만 validation을 5.546917까지 잃었다. 따라서 이번에는 regularization 강도를 늘리기보다 max_steps를 90에서 85로 아주 작게 줄여 optimization 길이만 조절하는 것이 가장 해석 가능하다.
- 바꾼 변수: `{"max_steps": 85}`
- 기대 결과: 성공 기준은 run073 대비 final_val_loss가 5.543 이하에 머물고, final_generalization_gap이 0.0 이하 또는 overfit_score가 run073의 0.015280보다 낮아지는 것이다. final_val_loss가 5.544 이상으로 악화되면 85 steps는 under-training 비용이 큰 것으로 본다. overfit_score가 그대로이거나 커지면 seed202 penalty는 학습 길이보다 activation/seed 특성에 가깝다고 본다.
- 실제 결과: final_val_loss=5.546870072682698, gap=-0.003162542978922822, overfit_score=0.007926543553669063, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0032, overfit_score=0.0079. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 max_steps=85가 seed202에서 validation을 유지하면서 overfit_score를 낮추면 같은 mish + ffn_mult=3 설정을 seed151에서 반복해 current best run072와 비교한다. seed151도 통과하면 seed134까지 확인해 max_steps=85가 3-seed 평균에서 90-step 후보를 대체할 수 있는지 판단한다. / 과적합 시 max_steps=85에서도 overfit_score가 줄지 않거나 validation이 악화되면 max_steps=90을 유지하고, 다음에는 mish seed202에서 weight_decay=0.015처럼 더 작은 regularization 단일축을 확인한다. weight_decay=0.02는 과거 high-overfit 구간에서 효과가 약했으므로 바로 크게 올리지는 않는다.

## run 083 - 2026-06-03T02:02:55+00:00

- 보고서: `docs/train/runs/run_083.md`
- 이번 가설: run082 showed that shortening mish + seed202 from 90 to 85 steps reduces overfit_score but loses too much validation performance, so the next safer test is to restore max_steps=90 and apply only a gentle weight_decay increase from 0.01 to 0.015. If the small seed202 penalty in run073 is due to train-side over-progress rather than activation choice, this should keep the low final_val_loss near 5.541 while reducing overfit_score without the under-training cost seen in run082.
- 근거: The activation sweep is effectively closed: mish has the best 3-seed mean final_val_loss among mish, silu, gelu_exact, and quick_gelu, and current best remains run072 with mish seed151. The matched seed202 mish baseline run073 had excellent final_val_loss=5.541102 but a small positive gap and overfit_score=0.015280. Attempts to reduce this penalty by after_activation dropout (run070), lower learning_rate (run071), and shorter max_steps=85 (run082) all worsened validation loss; run082 in particular improved overfit_score to 0.007927 but final_val_loss fell to 5.546870. A smaller weight_decay step is therefore the most conservative remaining regularization axis: it preserves architecture, activation, context/stride, learning rate, and 90-step optimization while testing whether mild optimizer regularization can reduce the seed202 penalty.
- 바꾼 변수: `{"max_steps": 90, "weight_decay": 0.015}`
- 기대 결과: Success means final_val_loss remains close to run073, ideally at or below 5.543, while overfit_score drops below 0.015280 and final_generalization_gap is near zero or negative. If final_val_loss rises toward run082 or run071 levels above 5.545, the seed202 penalty is not worth regularizing further and the loop should prefer the current mish 3-seed plateau.
- 실제 결과: final_val_loss=5.541118780771892, gap=0.0003363291422529002, overfit_score=0.015260616938273408, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0003, overfit_score=0.0153. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat the same mish + weight_decay=0.015 setting on seed151 to compare directly against current best run072; if seed151 remains competitive, complete seed134 for a 3-seed average. / 과적합 시 If weight_decay=0.015 does not reduce the seed202 penalty or worsens validation, stop tuning seed202 regularization and pivot to seed variance confirmation around the current best mish setting, or test a very small capacity/initialization axis such as init_std only if the dashboard still shows low-risk runs.

## run 084 - 2026-06-03T02:07:56+00:00

- 보고서: `docs/train/runs/run_084.md`
- 이번 가설: The seed202 regularization branch should be closed unless weight_decay=0.015 also improves the current best seed151 mish run. Run083 showed that weight_decay=0.015 preserved seed202 validation but did not meaningfully reduce its overfit-aware penalty versus run073, so applying the same mild weight decay to seed151 tests whether it is a real activation-setting improvement or just neutral noise. If seed151 keeps overfit_score at 0.0 and lowers final_val_loss below run072, weight_decay=0.015 can remain in the mish candidate; otherwise the loop should return to weight_decay=0.01 and treat the remaining differences as seed variance.
- 근거: Current best is run072: mish, seed151, ffn_mult=3, max_steps=90, weight_decay=0.01, final_val_loss=5.542158, final_generalization_gap=-0.017935, overfit_score=0.0. The matched seed202 baseline run073 had lower raw validation, final_val_loss=5.541102, but a small positive gap and overfit_score=0.015280. Attempts to reduce that seed202 penalty by shorter max_steps=85 lost too much validation in run082, while weight_decay=0.015 in run083 produced nearly identical validation and overfit_score to run073. Since seed151 already has no overfit penalty, the only useful reason to keep weight_decay=0.015 is if it improves or preserves the current best validation without harming the gap. This is a safe low-cost confirmation on the same MPS-balanced 413184-parameter setting.
- 바꾼 변수: `{"seed": 151, "weight_decay": 0.015}`
- 기대 결과: Success means final_val_loss is at or below run072's 5.542158 while final_generalization_gap stays negative and overfit_score remains 0.0. A neutral result near 5.5427 with overfit_score=0.0 means weight_decay=0.015 is harmless but not useful. If final_val_loss rises above 5.544 or overfit_score becomes positive, the regularization branch should be closed and mish weight_decay=0.01 should remain the default.
- 실제 결과: final_val_loss=5.542174339294434, gap=-0.01793646812438965, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0179, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If seed151 improves with weight_decay=0.015, repeat the same setting on seed134 to complete a 3-seed comparison against mish weight_decay=0.01 before changing the default. / 과적합 시 If seed151 does not improve, stop tuning weight_decay for this mish plateau and pivot to seed-variance-aware selection or a very small initialization/optimization axis only if the dashboard remains low risk.

## run 085 - 2026-06-03T02:12:49+00:00

- 보고서: `docs/train/runs/run_085.md`
- 이번 가설: After closing the weight_decay=0.015 regularization branch, the next most informative question is seed variance around the current best mish configuration. Keeping the run072 default candidate unchanged and changing only seed to 303 will show whether the mish + ffn_mult=3 + weight_decay=0.01 plateau remains low-risk outside the already-tested seeds 151, 202, and 134.
- 근거: The current best is run072 with mish, seed151, context_length=48, stride=24, ffn_mult=3, max_steps=90, weight_decay=0.01, final_val_loss=5.542158, final_generalization_gap=-0.017935, and overfit_score=0.0. The same mish setting already covered seed202 in run073 and seed134 in run074, giving a narrow but visible seed pattern: seed202 has the lowest raw validation but a small positive-gap penalty, while seed151 and seed134 have negative gaps and zero overfit_score. Attempts to improve the seed202 penalty through max_steps=85, weight_decay=0.015, and seed151 weight_decay confirmation did not beat run072. Therefore more regularization is low-value; a fresh seed is the cleanest way to determine whether the plateau is robust or whether current best selection is mostly seed variance.
- 바꾼 변수: `{"seed": 303}`
- 기대 결과: If the plateau is robust, final_val_loss should land in the existing mish 3-seed band around 5.541 to 5.549, fit_status should remain generalizing, and overfit_score should stay below 0.03. A result near or below run072 with overfit_score=0.0 would strengthen the case for mish as the default. A result above 5.552 or with a larger positive gap would indicate seed variance dominates the current tiny activation and regularization differences.
- 실제 결과: final_val_loss=5.559609095255534, gap=0.04233256975809763, overfit_score=0.15810060501098722, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0423, overfit_score=0.1581. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 If seed303 is low-risk and near the existing mish band, keep mish weight_decay=0.01 as the default and consider one more fresh seed only if the goal is confidence intervals rather than new best hunting. / 과적합 시 If seed303 shows a high positive gap or worse validation, stop local hyperparameter tweaks and summarize the plateau as seed-sensitive; the next useful direction would be broader data/window evaluation rather than more regularization.

## run 086 - 2026-06-03T02:18:53+00:00

- 보고서: `docs/train/runs/run_086.md`
- 이번 가설: Run085 showed that the current best mish configuration is seed-sensitive: seed303 drove train loss down to 5.517277 while validation stayed at 5.559609, producing a high positive gap and overfit_score=0.158101. Keeping the same seed303 and model settings but reducing stride from 24 to 16 will test whether denser overlapping training windows can reduce the train/validation mismatch without changing the Transformer architecture or activation. If the problem is window coverage sensitivity rather than model capacity alone, stride=16 should reduce the gap and overfit_score while keeping validation in a usable range.
- 근거: Earlier runs established context_length=48 with stride=24 as a strong data-window stabilizer: moving from default stride to stride=24 made runs 056-058 low-risk, while context_length=64 and stride=32 failed in run059. The recent activation and regularization plateau from runs 072-084 showed tiny differences among mish, gelu_exact, silu, quick_gelu, weight_decay, and step-count tweaks. Run085 is different: changing only seed to 303 created an overfit_risk result with final_generalization_gap=0.042333 and train_val_improvement_gap=0.057884. Rather than another weight_decay or activation tweak, the next highest-information safe step is to keep seed303 fixed and adjust only stride to see if a denser windowing condition stabilizes the bad seed.
- 바꾼 변수: `{"stride": 16}`
- 기대 결과: Success means final_generalization_gap falls well below run085's 0.042333, overfit_score drops below 0.08, and final_val_loss stays below about 5.56. A strong success would restore low-risk generalizing with final_val_loss inside the prior mish band near 5.541-5.549. If stride=16 worsens validation or keeps overfit_score high, denser overlap is not the right fix and the seed303 issue should be treated as broader split/seed sensitivity.
- 실제 결과: final_val_loss=5.5559156735738116, gap=0.013590892155964873, overfit_score=0.0657787322998038, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0136, overfit_score=0.0658. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If stride=16 stabilizes seed303, repeat stride=16 on one known-good seed such as 151 to verify it does not degrade the current best band before considering it as a robustness default. / 과적합 시 If stride=16 does not reduce the seed303 gap, stop local hyperparameter polishing and move to broader seed/window evaluation, such as comparing multiple fresh seeds or documenting that current best is seed-sensitive under this corpus split.

## run 087 - 2026-06-03T02:24:43+00:00

- 보고서: `docs/train/runs/run_087.md`
- 이번 가설: Run086 showed that stride=16 can rescue the bad seed303 case by reducing the final gap from 0.042333 to 0.013591 and the overfit_score from 0.158101 to 0.065779, but its final_val_loss remained far above the current best. The next question is whether stride=16 is a generally useful robustness default or only a rescue knob for seed303. Repeating stride=16 on the known-good best seed151, while keeping the mish ffn_mult=3 model and optimizer fixed, will test whether denser windows preserve the run072 best band or trade away validation quality.
- 근거: The current best remains run072 with mish, seed151, context_length=48, stride=24, final_val_loss=5.542158, final_generalization_gap=-0.017935, overfit_score=0.0, and parameter_count=413184. Recent activation, step-count, and weight_decay experiments produced only tiny gains or regressions, while run085 exposed substantial seed/window sensitivity. Run086 improved the seed303 overfit signals without changing architecture, suggesting stride is the highest-information axis. Before adopting stride=16, it must be checked against the strongest matched seed151 baseline because a robustness default that degrades the best band is not worth taking.
- 바꾼 변수: `{"stride": 16}`
- 기대 결과: If stride=16 is a real robustness default, seed151 should stay in the current best band with final_val_loss near 5.542-5.544, overfit_score near 0.0, and fit_status=generalizing. If final_val_loss rises above about 5.545 while overfit_score remains low, stride=16 should be treated as a seed303 rescue setting rather than the default. If gap or overfit_score rises, denser overlap is harmful for known-good seeds.
- 실제 결과: final_val_loss=5.547791798909505, gap=-0.033585826555888154, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0336, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If seed151 with stride=16 stays at or below the run072 band, repeat stride=16 on seed202 or seed134 to estimate whether it improves average robustness across the existing mish seeds. / 과적합 시 If stride=16 creates a positive gap or worse overfit_score on seed151, return to stride=24 as the default and document stride=16 as a targeted rescue for seed303 only.

## run 088 - 2026-06-03T02:28:55+00:00

- 보고서: `docs/train/runs/run_088.md`
- 이번 가설: Run087 showed that stride=16 is not a good general default: on the known-good seed151 it kept overfit_score at 0.0 but raised final_val_loss to 5.547792, well above the run072 best band. Therefore the loop should return to the run072 mish + ffn_mult=3 + stride=24 default and measure fresh seed variance directly. Running a new seed404 with the unchanged default will test whether run085 seed303 was an isolated bad seed/window draw or whether the current best plateau is broadly seed-sensitive.
- 근거: The current best remains run072 with seed151, final_val_loss=5.542158, final_generalization_gap=-0.017935, overfit_score=0.0, parameter_count=413184, and MPS throughput around 29916 tokens/sec. The matched mish defaults across seed202 and seed134 were still low-risk, but seed303 at stride=24 became overfit_risk with final_val_loss=5.559609 and overfit_score=0.158101. Stride=16 rescued the seed303 gap in run086 but did not restore best-band validation, and run087 showed it also hurts a known-good seed. More activation, weight_decay, or step-count polishing has already produced tiny changes, so the highest-information safe next step is a fresh default seed rather than another hyperparameter tweak.
- 바꾼 변수: `{"seed": 404}`
- 기대 결과: If seed303 was an outlier, seed404 should land near the established mish band with final_val_loss below about 5.55, fit_status=generalizing, and overfit_score below 0.03. If seed404 again rises above 5.552 or develops a positive gap above 0.03, then seed variance is a core limitation of the current split/window setup and the next direction should be broader seed/window evaluation rather than local tuning.
- 실제 결과: final_val_loss=5.548481464385986, gap=0.0577845573425293, overfit_score=0.17657971382141113, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0578, overfit_score=0.1766. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 If seed404 is low-risk and near the existing mish band, keep stride=24 as the default and run one more fresh seed only to estimate confidence intervals, not to chase a single-run best. / 과적합 시 If seed404 repeats the seed303 overfit pattern, stop activation and regularization polishing and compare data-window conditions across multiple fresh seeds, starting with whether stride=16 consistently reduces gap while accepting its validation cost.

## run 089 - 2026-06-03T02:33:38+00:00

- 보고서: `docs/train/runs/run_089.md`
- 이번 가설: Run088 confirmed that the current mish + ffn_mult=3 default is broadly seed-sensitive: fresh seed404 at stride=24 drove train loss down to 5.490697 while validation stayed at 5.548481, producing final_generalization_gap=0.057785 and overfit_score=0.176580. Run086 showed that reducing stride from 24 to 16 rescued a similar bad fresh seed303 gap, while run087 showed stride=16 is not a good global default for seed151. Keeping seed404 fixed and changing only stride to 16 will test whether stride=16 is a targeted rescue window for overfitting fresh seeds rather than a default setting.
- 근거: The dashboard now has two fresh-seed overfit cases at the default stride=24: seed303 in run085 and seed404 in run088. Both lowered train loss aggressively while validation lagged. For seed303, the matched stride=16 run086 reduced overfit_score from 0.158101 to 0.065779 and brought fit_status back to generalizing, but validation stayed worse than best. For known-good seed151, run087 showed stride=16 kept overfit_score=0.0 but raised final_val_loss above the best band, so it should not replace stride=24 globally. The highest-information next step is therefore a matched seed404 stride rescue test, not more activation, regularization, or optimizer polishing.
- 바꾼 변수: `{"stride": 16}`
- 기대 결과: Success means stride=16 reduces seed404 final_generalization_gap well below run088's 0.057785 and overfit_score below about 0.08, ideally restoring fit_status=generalizing while keeping final_val_loss no worse than about 5.56. If validation rises substantially or overfit_score remains high, stride=16 is not a reliable rescue and the loop should move to broader seed/window or data split evaluation.
- 실제 결과: final_val_loss=5.555461088816325, gap=0.0183949867884321, overfit_score=0.03291022777557462, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0184, overfit_score=0.0329. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If stride=16 also rescues seed404's gap, document stride=16 as an overfit-seed rescue setting and compare one more fresh overfitting seed or a smaller max_steps/window condition to find a better validation/robustness tradeoff. / 과적합 시 If stride=16 fails on seed404, stop local stride polishing and test a different robustness axis such as shorter max_steps on the fresh overfit seeds, while keeping the architecture fixed.

## run 090 - 2026-06-03T02:39:29+00:00

- 보고서: `docs/train/runs/run_090.md`
- 이번 가설: Run089 confirmed that stride=16 is a reliable rescue knob for fresh overfit seeds, but it pays for lower overfit with worse validation: seed404 moved from run088 final_val_loss=5.548481 and overfit_score=0.176580 to run089 final_val_loss=5.555461 and overfit_score=0.032910. A shorter training horizon at the original stride=24 may keep more of run088's validation advantage while avoiding the severe train-side over-progress. Keeping seed404 fixed, restoring stride=24, and reducing max_steps from 90 to 80 will test whether early stopping is a better validation/robustness tradeoff than denser windows.
- 근거: The current best run072 remains seed151, stride=24, max_steps=90, final_val_loss=5.542158, gap=-0.017935, overfit_score=0.0. Fresh seeds 303 and 404 both overfit at stride=24/max_steps=90, while stride=16 rescued their gaps but produced validation around 5.556. The run088 seed404 failure is especially diagnostic because train_loss fell to 5.490697 while validation stayed at 5.548481. That looks like optimization over-progress rather than insufficient model capacity. Since previous activation and weight_decay tweaks are exhausted, the next small safe optimization test is to stop earlier at 80 steps with the original data window.
- 바꾼 변수: `{"max_steps": 80, "stride": 24}`
- 기대 결과: A useful tradeoff would keep final_val_loss closer to run088 than run089, ideally below 5.552, while reducing final_generalization_gap far below 0.057785 and overfit_score below about 0.08. If max_steps=80 still overfits, then stopping at 80 is insufficient. If validation worsens toward stride16 levels without much overfit benefit, then the loop should test either max_steps=70 or a learning_rate reduction only if it can preserve validation better.
- 실제 결과: final_val_loss=5.559826691945394, gap=0.05291338761647513, overfit_score=0.16196620464324862, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0529, overfit_score=0.1620. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 If max_steps=80 gives lower validation than stride16 while keeping overfit_score low, repeat the same shorter horizon on seed303 to verify it is a general fresh-seed rescue setting. / 과적합 시 If max_steps=80 keeps overfit_score high, try a stronger early-stop point such as max_steps=70 on the same seed404 before changing architecture or activation.

## run 091 - 2026-06-03T02:44:43+00:00

- 보고서: `docs/train/runs/run_091.md`
- 이번 가설: For the seed404 mish baseline that overfit at stride 24, reducing learning_rate from 0.0003 to 0.000275 will slow train-side over-progress enough to reduce the generalization gap without the validation-loss penalty seen from stride16 or max_steps80.
- 근거: Runs 088-090 isolate seed404 as a useful stress case: default stride24 overfit with final_val_loss 5.548481 and overfit_score 0.176580, stride16 rescued the gap but worsened validation to 5.555461, and max_steps80 kept stride24 but worsened validation further to 5.559827 while still overfitting. A small learning-rate reduction keeps the data geometry and training budget fixed, directly testing whether optimization pace can rescue the seed before using stride16 as the default rescue knob.
- 바꾼 변수: `{"learning_rate": 0.000275}`
- 기대 결과: Success means final_val_loss stays near the seed404 default run rather than the stride16 or max_steps80 penalty, ideally <= 5.552, while overfit_score drops clearly below 0.176580 and preferably below 0.08. If validation rises toward 5.556+ or overfit_score remains high, learning-rate reduction alone is not a good rescue.
- 실제 결과: final_val_loss=5.5547966957092285, gap=0.05480945110321045, overfit_score=0.1676543951034546, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0548, overfit_score=0.1677. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 Repeat the 0.000275 learning-rate rescue on seed303 to test whether it generalizes across overfit-prone seeds before changing the default plan. / 과적합 시 Keep stride16 as the confirmed rescue knob for high-gap seeds, and test a small regularization-side change such as drop_rate 0.14 on seed404 while preserving stride24 and max_steps90.

## run 092 - 2026-06-03T02:48:48+00:00

- 보고서: `docs/train/runs/run_092.md`
- 이번 가설: For the seed404 mish baseline that keeps overfitting at stride 24, a mild drop_rate increase from 0.12 to 0.14 will reduce train-side over-progress and overfit_score more effectively than the failed learning-rate and early-stop probes, while preserving validation better than the stride16 rescue.
- 근거: The recent seed404 sequence isolates the tradeoff. Run088 at the default stride24 reached final_val_loss 5.548481 but overfit_score 0.176580. Run089 with stride16 lowered overfit_score to 0.032910 but worsened final_val_loss to 5.555461, making it a costly rescue rather than a default. Run090 max_steps80 and run091 learning_rate 0.000275 both stayed overfit_risk with worse validation than run088. A small dropout increase keeps the same data geometry, optimizer, capacity, and 90-step budget while testing whether regularization can close the gap without the stride16 validation penalty.
- 바꾼 변수: `{"drop_rate": 0.14}`
- 기대 결과: A useful result would keep final_val_loss below the stride16 rescue band, ideally <= 5.552, while reducing overfit_score meaningfully below run088 and run091, preferably below 0.10. If validation rises toward 5.556+ or overfit_score remains above about 0.15, mild dropout is not enough for seed404.
- 실제 결과: final_val_loss=5.548829714457194, gap=0.0565950075785322, overfit_score=0.17301106452941983, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0566, overfit_score=0.1730. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 Repeat drop_rate 0.14 on seed303 with the same mish stride24 default to test whether mild dropout generalizes across the two overfit-prone fresh seeds. / 과적합 시 Treat mild dropout as insufficient and compare the confirmed stride16 rescue against a slightly stronger regularization setting such as drop_rate 0.16 on seed404, while watching for validation-loss drift.

## run 093 - 2026-06-03T02:53:58+00:00

- 보고서: `docs/train/runs/run_093.md`
- 이번 가설: For the seed404 mish baseline, increasing drop_rate from 0.14 to 0.16 may cross the regularization threshold needed to reduce the high stride24 overfit_score while preserving the validation advantage over the stride16 rescue.
- 근거: Run088 showed the default seed404 stride24 configuration has useful validation loss at 5.548481 but a high overfit_score of 0.176580. Run089 proved stride16 can rescue the gap, lowering overfit_score to 0.032910, but it worsened validation to 5.555461. Run090 early stopping and run091 learning-rate reduction both failed to improve the tradeoff. Run092 drop_rate 0.14 kept validation close to default at 5.548830 but left overfit_score high at 0.173011. A slightly stronger drop_rate 0.16 is still a small single-axis regularization change, runs quickly on the MPS profile, and tests whether dropout strength has a threshold effect before abandoning this branch.
- 바꾼 변수: `{"drop_rate": 0.16}`
- 기대 결과: This run is useful if final_val_loss remains below the stride16 rescue band, ideally <= 5.552, while overfit_score drops meaningfully below run092 and preferably below 0.12. If final_val_loss drifts toward 5.556+ or overfit_score remains above 0.15, stronger dropout is not a good seed404 rescue.
- 실제 결과: final_val_loss=5.549636999766032, gap=0.05571929613749216, overfit_score=0.17038393020629972, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0557, overfit_score=0.1704. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 Repeat drop_rate 0.16 on seed303 with stride24 and max_steps90 to see whether stronger dropout rescues both known overfit-prone fresh seeds. / 과적합 시 Close the dropout-strength branch and return to context/stride experiments: either keep stride16 as a targeted rescue or test a less costly window change such as stride20 on seed404.

## run 094 - 2026-06-03T02:58:34+00:00

- 보고서: `docs/train/runs/run_094.md`
- 이번 가설: For the seed404 mish baseline, reducing stride from 24 to 20 will add moderate window overlap that lowers the overfit gap more than the default stride24 run, while avoiding the larger validation-loss penalty observed at stride16.
- 근거: The seed404 branch has isolated stride/windowing as the only axis that clearly rescues overfit. Run088 with stride24 had good validation loss at 5.548481 but a high overfit_score of 0.176580. Run089 with stride16 reduced overfit_score to 0.032910 and changed fit_status to generalizing, but validation worsened to 5.555461. Attempts to preserve stride24 through max_steps80, learning_rate 0.000275, drop_rate 0.14, and drop_rate 0.16 all remained overfit_risk with overfit_score around 0.16-0.17. Testing stride20 restores the default regularization and probes whether an intermediate amount of overlap gives a better validation/generalization tradeoff than either endpoint.
- 바꾼 변수: `{"stride": 20}`
- 기대 결과: Success means final_val_loss stays closer to the default stride24 band than the stride16 rescue, ideally <= 5.552, while overfit_score drops clearly below the stride24/dropout branch and preferably below 0.10. If overfit_score stays above 0.15, stride20 is not enough; if final_val_loss drifts toward 5.555+, it behaves too much like stride16.
- 실제 결과: final_val_loss=5.543789545694987, gap=-0.03905232747395804, overfit_score=0.0, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0391, overfit_score=0.0000. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat stride20 on seed303 to check whether the intermediate-overlap rescue generalizes across both known overfit-prone fresh seeds; then test seed151 only if the rescue is robust. / 과적합 시 If stride20 fails, close the intermediate stride branch and keep stride16 as the targeted rescue, or test a slightly denser stride18 only if validation remains substantially better than stride16.

## run 095 - 2026-06-03T03:03:40+00:00

- 보고서: `docs/train/runs/run_095.md`
- 이번 가설: The stride20 intermediate-overlap setting that rescued seed404 will also rescue the earlier seed303 overfit case, lowering overfit_score versus stride24 while avoiding the validation-loss penalty seen at stride16.
- 근거: Run094 showed a strong seed404 result: stride20 changed the branch from repeated overfit_risk to generalizing with final_val_loss 5.543790, final_generalization_gap -0.039052, and overfit_score 0.0. The earlier seed303 branch has the same pattern to test against: run085 at stride24 overfit with final_val_loss 5.559609 and overfit_score 0.158101, while run086 at stride16 reduced the gap but still landed at final_val_loss 5.555916 and overfit_score 0.065779. Repeating stride20 on seed303 directly tests whether intermediate window overlap is a robust rescue knob across both known overfit-prone fresh seeds rather than a seed404-specific accident.
- 바꾼 변수: `{"seed": 303}`
- 기대 결과: Success means seed303 at stride20 improves on both seed303 comparison points: final_val_loss below the stride16 rescue band, ideally <= 5.552, and overfit_score below run086's 0.065779 or at least far below run085's 0.158101. If validation stays around 5.556+ or overfit_score remains above 0.10, stride20 is less robust than the seed404 result suggested.
- 실제 결과: final_val_loss=5.549335797627767, gap=0.008998235066731475, overfit_score=0.053554296493529385, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0090, overfit_score=0.0536. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Test stride20 on the known-good seed151 to ensure it does not trade away the run072 best-band validation when used beyond overfit-prone seeds. / 과적합 시 If seed303 is not rescued, treat stride20 as seed404-specific and keep stride16 as the more reliable targeted rescue; optionally test stride18 only if validation remains clearly better than stride16.

## run 096 - 2026-06-03T03:10:45+00:00

- 보고서: `docs/train/runs/run_096.md`
- 이번 가설: The stride20 intermediate-overlap setting that rescued seed404 and seed303 will preserve the known-good seed151 validation band, showing it can be considered a robust default candidate rather than only an overfit rescue knob.
- 근거: Run094 changed seed404 from repeated stride24 overfit_risk to generalizing with final_val_loss 5.543790 and overfit_score 0.0, and run095 repeated the stride20 rescue on seed303 with final_val_loss 5.549336 and overfit_score 0.053554, beating the seed303 stride16 validation band. The remaining risk is that stride20 may help overfit-prone seeds by adding overlap while slightly hurting the best seed151 baseline: run072 at stride24 remains best with final_val_loss 5.542158 and overfit_score 0.0, while run087 at stride16 stayed generalizing but rose to final_val_loss 5.547792. Testing stride20 on seed151 directly separates robust-default evidence from rescue-only evidence.
- 바꾼 변수: `{"seed": 151, "stride": 20}`
- 기대 결과: Success means seed151 at stride20 stays in the run072 best band: final_val_loss ideally <= 5.544 with final_generalization_gap near or below zero and overfit_score near 0.0. If validation rises toward the stride16 seed151 result around 5.5478, stride20 should be treated as a rescue setting rather than a replacement default.
- 실제 결과: final_val_loss=5.547611077626546, gap=0.02041419347127249, overfit_score=0.07020362218220999, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0204, overfit_score=0.0702. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Test stride20 on another historically strong seed such as seed202, then compare the seed151/202/303/404 stride20 mean against the stride24 and stride16 branches before promoting stride20 as the next default. / 과적합 시 Keep stride24 as the default for known-good seeds and reserve stride20 for overfit-prone seeds; if seed151 degrades but does not overfit, consider a narrower stride18 rescue probe only on bad seeds.

## run 097 - 2026-06-03T03:13:41+00:00

- 보고서: `docs/train/runs/run_097.md`
- 이번 가설: For the overfit-prone seed303 mish configuration, reducing stride from 20 to 18 will add a little more overlap that lowers the remaining positive generalization gap and overfit_score while preserving most of stride20's validation advantage over the stride16 rescue.
- 근거: The recent context/stride branch now separates default and rescue behavior. Seed303 at stride24 overfit in run085 with final_val_loss 5.559609 and overfit_score 0.158101. Stride16 rescued the gap in run086 but still landed at final_val_loss 5.555916. Stride20 improved the tradeoff in run095 with final_val_loss 5.549336 and overfit_score 0.053554, but it still had a positive gap. Run096 then showed stride20 is not a global default because seed151 rose to final_val_loss 5.547611 versus the run072 best 5.542158. Therefore the next safe, hardware-light test is a rescue-only stride18 probe on seed303: denser than stride20 to target the remaining gap, but less dense than stride16 to avoid the validation penalty.
- 바꾼 변수: `{"seed": 303, "stride": 18}`
- 기대 결과: A useful rescue would keep final_val_loss below the stride16 seed303 result and close to the stride20 result, ideally <= 5.552, while lowering overfit_score below run095's 0.053554 or at least keeping it clearly below run085's 0.158101. If validation drifts toward 5.556 or the gap stays near the stride20 level, stride18 adds cost without improving the rescue.
- 실제 결과: final_val_loss=5.548077901204427, gap=0.022380193074543975, overfit_score=0.09365868568420321, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0224, overfit_score=0.0937. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat stride18 on seed404 to see whether it preserves the strong run094 validation while confirming that stride18 is a robust targeted rescue setting for overfit-prone seeds. / 과적합 시 Keep stride20 as the preferred rescue point for seed303 and seed404, and stop denser stride polishing unless a new overfit seed shows a different failure pattern.

## run 098 - 2026-06-03T03:19:17+00:00

- 보고서: `docs/train/runs/run_098.md`
- 이번 가설: A fresh seed505 run with the current mish stride24 default will clarify whether the recent seed303 and seed404 overfit cases are common seed variance or isolated bad draws; if seed505 is low-risk, stride20 should remain a targeted rescue rather than the main default.
- 근거: The latest stride branch reached a clear stopping point. Runs085 and 088 showed that the mish + ffn_mult=3 + stride24 default can overfit badly on fresh seeds303 and 404, but run094 and run095 showed stride20 is an effective targeted rescue for those overfit-prone seeds. Run096 showed stride20 is not a global default because seed151 lost the run072 best band, and run097 showed denser stride18 worsened seed303's overfit signal versus stride20. The highest-information next step is therefore not more stride polishing, but a fresh stride24 seed-variance probe. This keeps the proven run072 configuration intact and asks whether the default remains acceptable on a new seed before spending more loops on rescue variants.
- 바꾼 변수: `{"seed": 505}`
- 기대 결과: If the current default is robust enough, seed505 should finish generalizing with final_val_loss in or near the established mish band, ideally <= 5.548, final_generalization_gap below about 0.03, and overfit_score below 0.10. If it repeats the seed303/404 pattern with gap above 0.04 or overfit_score above 0.15, the loop should treat fresh-seed overfit as a common failure mode and immediately test the targeted stride20 rescue on the same seed.
- 실제 결과: final_val_loss=5.552542209625244, gap=0.02306497097015381, overfit_score=0.02306497097015381, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0231, overfit_score=0.0231. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Run one more fresh seed at stride24 only if confidence intervals are still unclear; otherwise consolidate the policy as stride24 default plus stride20 rescue for high-gap seeds. / 과적합 시 Keep seed505 fixed and test stride20 with the same mish configuration, since stride20 has beaten stride16 and stride18 as the cleanest rescue tradeoff so far.

## run 099 - 2026-06-03T03:24:18+00:00

- 보고서: `docs/train/runs/run_099.md`
- 이번 가설: A second fresh stride24 default probe with seed606 will show whether seed505's low-risk but higher-validation result is typical variance, while checking whether high-gap failures like seed303 and seed404 are frequent enough to make stride20 rescue a standard follow-up policy.
- 근거: The recent loop has separated three cases: strong default seeds such as run072 seed151, high-gap default failures such as run085 seed303 and run088 seed404, and low-risk but higher-validation variance in run098 seed505. Stride20 rescued the high-gap seeds in runs094 and 095, but run096 showed it should not replace stride24 globally, and run097 showed denser stride18 is worse than stride20 for seed303. Since seed505 did not have a high overfit_score, immediately testing stride20 on it would blur the rescue rule. One more fresh stride24 seed is the safest small experiment to estimate whether the mish default usually generalizes acceptably or whether fresh-seed validation variance itself is the next bottleneck.
- 바꾼 변수: `{"seed": 606}`
- 기대 결과: A stable default result would be generalizing with final_val_loss near the mish plateau, ideally <= 5.550, final_generalization_gap below 0.03, and overfit_score below 0.10. If seed606 has gap above 0.04 or overfit_score above 0.15, the next run should keep seed606 fixed and test stride20 as the confirmed targeted rescue. If it is low-risk but validation remains above 5.552 like seed505, the next direction should shift from overfit rescue to reducing seed-level validation variance.
- 실제 결과: final_val_loss=5.542599201202393, gap=0.0032129287719726562, overfit_score=0.06227644284566303, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0032, overfit_score=0.0623. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Consolidate the current operating policy as stride24 default plus stride20 rescue only for high-gap seeds, then move to a variance-reduction axis such as modest capacity or context/stride comparison only if more fresh seeds remain high-validation. / 과적합 시 Keep seed606 fixed and test stride20 with the same mish configuration, because stride20 is currently the cleanest high-gap rescue compared with stride16, stride18, early stopping, learning-rate reduction, and dropout increases.

## run 100 - 2026-06-03T03:29:36+00:00

- 보고서: `docs/train/runs/run_100.md`
- 이번 가설: For the strong seed606 stride24 default run, lowering init_std from 0.02 to 0.018 will reduce seed-level optimization variance and the remaining overfit_score without changing model architecture, data windowing, or training length.
- 근거: Run099 showed that the current mish + ffn_mult=3 + stride24 default can still land near the best band on a fresh seed: seed606 reached final_val_loss 5.542599 with final_generalization_gap 0.003213. This supports the operating policy of stride24 as default and stride20 as a targeted rescue for high-gap seeds such as seed303 and seed404. The remaining question is not more stride polishing, but whether a tiny initialization-scale change can reduce run-to-run variance and overfit_score while preserving the near-best validation band. Prior max_steps, weight_decay, dropout, learning-rate, and stride variants either failed to improve the best band or were rescue-only. Lowering init_std slightly is a safe, single-axis optimization/initialization probe that keeps parameter_count and runtime unchanged.
- 바꾼 변수: `{"init_std": 0.018}`
- 기대 결과: A useful result would keep seed606 in the best band, ideally final_val_loss <= 5.543, while lowering final_generalization_gap toward zero and reducing overfit_score below run099's 0.062276. If validation rises above about 5.546 or overfit_score increases, init_std=0.018 is not a good default-side variance reducer and the loop should keep init_std=0.02.
- 실제 결과: final_val_loss=5.5438259442647295, gap=0.0019657214482622365, overfit_score=0.05373672644297134, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0020, overfit_score=0.0537. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat init_std=0.018 on seed151 or seed202 to test whether the initialization-scale improvement transfers to established best-band seeds before considering it as a new default. / 과적합 시 Revert to init_std=0.02 and keep the current policy: stride24 default, stride20 only for high-gap rescue. Avoid further initialization tuning unless another fresh seed shows the same near-best but positive overfit_score pattern.

## run 101 - 2026-06-03T03:34:01+00:00

- 보고서: `docs/train/runs/run_101.md`
- 이번 가설: For the strong seed606 mish stride24 default run, increasing max_steps from 90 to 100 will give the near-best validation trajectory enough extra optimization to beat run072 while keeping the final gap low.
- 근거: Run099 showed that the current mish + ffn_mult=3 + stride24 default remains a strong default on a fresh seed: seed606 reached final_val_loss 5.542599, only about 0.00044 above the best run072, with final_generalization_gap 0.003213. Run100 lowered init_std to 0.018 and reduced the gap slightly, but worsened validation to 5.543826, so the initialization branch should not be promoted. Since stride20 is now a targeted high-gap rescue rather than a default, and stronger regularization or shorter runs have previously lost validation, the next most direct small test is optimization length: keep the successful seed606/default geometry and train only 10 extra steps. This tests whether the near-best run099 was still mildly optimization-limited or already at the overfit boundary.
- 바꾼 변수: `{"max_steps": 100}`
- 기대 결과: A useful result would lower final_val_loss below run099's 5.542599 and ideally below run072's 5.542158, while keeping final_generalization_gap below 0.02 and overfit_score below 0.08. If train loss improves but validation rises or overfit_score climbs above 0.10, max_steps=100 is beyond the useful optimization boundary and the default should remain max_steps=90.
- 실제 결과: final_val_loss=5.530441125233968, gap=0.01611717542012503, overfit_score=0.10098918279012015, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0161, overfit_score=0.1010. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat max_steps=100 on seed151 or seed202 to test whether the longer optimization horizon transfers beyond seed606 before considering it as a default. / 과적합 시 Revert to max_steps=90 and keep the current policy: stride24 default at init_std 0.02, stride20 only for high-gap rescue. Move next to a different variance-reduction axis rather than training longer.

## run 102 - 2026-06-03T03:39:22+00:00

- 보고서: `docs/train/runs/run_102.md`
- 이번 가설: Repeating the max_steps=100 horizon on the known best-band seed151 mish stride24 configuration will show whether run101's large validation gain was a transferable optimization effect rather than a seed606-specific lucky trajectory.
- 근거: Run101 lowered raw final_val_loss dramatically to 5.530441 by increasing max_steps from 90 to 100 on seed606, but its overfit_score rose to 0.100989 and the dashboard still treats run072 as the best overfit-aware candidate. Run072 remains the clean reference point: seed151, mish, stride24, max_steps90, final_val_loss 5.542158, final_generalization_gap -0.017935, overfit_score 0.0. Testing seed151 at max_steps100 is the highest-information follow-up because it directly checks whether the extra 10 steps improve the established best seed or merely over-optimize train loss on a favorable seed.
- 바꾼 변수: `{"max_steps": 100, "seed": 151}`
- 기대 결과: A successful transfer result should lower seed151 validation below the run072 baseline of 5.542158, ideally without pushing final_generalization_gap above 0.02 or overfit_score above 0.08. If train loss improves but validation stays flat or overfit_score reaches the medium-risk band around 0.10, max_steps=100 should remain a seed606 observation rather than a default.
- 실제 결과: final_val_loss=5.534507115681966, gap=-0.0005327860514325877, overfit_score=0.011693954467773438, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=-0.0005, overfit_score=0.0117. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 Repeat max_steps=100 on seed202 to estimate whether the longer horizon improves more than one historically strong seed before promoting it as the new stride24 default. / 과적합 시 Revert the default horizon to max_steps=90 and keep the current policy: mish stride24 at init_std 0.02 as default, stride20 only for high-gap rescue. Move next to a different variance-reduction axis instead of training longer.

## run 103 - 2026-06-03T03:43:56+00:00

- 보고서: `docs/train/runs/run_103.md`
- 이번 가설: Running the current mish stride24 max_steps=100 candidate on seed202 will test whether the longer optimization horizon is robust across another historically strong low-risk seed before making it the new default.
- 근거: Run102 promoted max_steps=100 from a seed606 observation into the current overfit-aware best on seed151, reaching final_val_loss 5.534507 with final_generalization_gap -0.000533 and overfit_score 0.011694. Earlier seed202 evidence was also strong at the mish stride24 baseline: the max_steps90 branch stayed low-risk, and weight_decay/shorter-step attempts did not justify a different regularization or under-training policy. The key question now is whether max_steps=100 improves seed202 without recreating the medium-risk signal seen in run101. A seed202 transfer check is safer and more informative than changing capacity, stride, dropout, or activation while the optimization-horizon hypothesis is still being validated.
- 바꾼 변수: `{"max_steps": 100, "seed": 202}`
- 기대 결과: A successful result should keep fit_status=generalizing, improve or at least match the historical seed202 mish band around final_val_loss 5.541, and keep final_generalization_gap below about 0.02 with overfit_score below 0.08. If validation worsens toward 5.546+ or overfit_score rises into medium-risk territory, max_steps=100 should remain promising but not yet a universal default.
- 실제 결과: final_val_loss=5.528694152832031, gap=0.008664369583129883, overfit_score=0.040244738260904356, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0087, overfit_score=0.0402. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If seed202 also benefits from max_steps=100, promote mish stride24 max_steps100 as the default candidate and test one fresh seed such as seed707 to measure seed variance under the new horizon. / 과적합 시 If seed202 overfits or loses validation quality, keep run102 as the best candidate but avoid broad promotion. Compare max_steps95 on seed202 or return to max_steps90 for historically stable seeds while preserving stride20 only as the high-gap rescue.

## run 104 - 2026-06-03T03:49:08+00:00

- 보고서: `docs/train/runs/run_104.md`
- 이번 가설: Using the promoted mish stride24 max_steps=100 candidate on a fresh seed707 will measure whether the longer-horizon improvement generalizes beyond the already-tested strong seeds or exposes renewed seed variance.
- 근거: Runs101-103 shifted the main hypothesis from activation or stride tuning to optimization horizon. Run101 showed that max_steps=100 can sharply lower raw validation on seed606 but with medium overfit risk. Run102 transferred the same horizon to the known best-band seed151 and became the overfit-aware best with final_val_loss 5.534507, final_generalization_gap -0.000533, and overfit_score 0.011694. Run103 then improved raw validation further on seed202 to 5.528694 while staying low-risk, although the overfit-aware dashboard still prefers run102. Since the earlier weakness of the mish stride24 setup was seed variance, the next safest research step is not another local tuning branch but a fresh seed probe under the new max_steps=100 candidate.
- 바꾼 변수: `{"max_steps": 100, "seed": 707}`
- 기대 결과: A robust default candidate should finish generalizing with final_val_loss in the new 100-step band, ideally below 5.545, final_generalization_gap below 0.03, and overfit_score below 0.10. A particularly strong result would approach run102/run103 without medium-risk overfit. If seed707 repeats the older high-gap seed303/404 behavior, the longer horizon improves strong seeds but still needs a rescue policy for unlucky seeds.
- 실제 결과: final_val_loss=5.533458232879639, gap=0.0504918098449707, overfit_score=0.2164137363433838, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0505, overfit_score=0.2164. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 If seed707 is low-risk, keep mish stride24 max_steps100 as the default candidate and run one more fresh seed or summarize the new policy as max_steps100 default plus stride20 rescue only for high-gap failures. / 과적합 시 If seed707 overfits, keep max_steps100 for strong seeds but immediately test the established stride20 rescue on the same seed707 before changing activation, dropout, capacity, or learning rate.

## run 105 - 2026-06-03T03:54:02+00:00

- 보고서: `docs/train/runs/run_105.md`
- 이번 가설: For the overfit-prone fresh seed707 under the promoted max_steps=100 candidate, reducing stride from 24 to 20 will add moderate window overlap that lowers the high generalization gap while preserving most of the strong raw validation.
- 근거: Run104 showed that mish stride24 max_steps100 can still fail by seed: seed707 reached competitive final_val_loss 5.533458, but train loss fell to 5.482966, producing final_generalization_gap 0.050492 and overfit_score 0.216414. Earlier stride experiments established stride20 as the cleanest targeted rescue for high-gap seeds: it rescued seed404 from overfit_risk to generalizing with overfit_score 0.0 and improved seed303 versus both stride24 and stride16, while stride20 was not good enough to become a global default for seed151. Therefore the safest next test is not activation, dropout, capacity, or learning-rate tuning, but a matched seed707 stride20 rescue while keeping the new 100-step horizon intact.
- 바꾼 변수: `{"max_steps": 100, "seed": 707, "stride": 20}`
- 기대 결과: A useful rescue should reduce seed707 final_generalization_gap well below 0.050492 and overfit_score below 0.10, ideally restoring fit_status=generalizing while keeping final_val_loss near the new 100-step band and below about 5.545. If validation worsens toward 5.55+ or overfit_score remains high, stride20 is not enough under the longer horizon.
- 실제 결과: final_val_loss=5.547992706298828, gap=0.01609349250793457, overfit_score=0.01609349250793457, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0161, overfit_score=0.0161. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If stride20 rescues seed707 under max_steps100, document the policy as mish max_steps100 default for low-risk seeds plus stride20 rescue for high-gap fresh seeds, then test one more fresh seed only if more variance confidence is needed. / 과적합 시 If stride20 still overfits seed707, test whether lowering max_steps to 95 on seed707 stride20 reduces train-side over-progress before changing regularization, capacity, activation, or learning rate.

## run 106 - 2026-06-03T04:00:28+00:00

- 보고서: `docs/train/runs/run_106.md`
- 이번 가설: For the overfit-prone seed707 max_steps=100 case, reducing stride from 24 to 20 will lower the high generalization gap while preserving the original 413k-parameter mish architecture and default regularization.
- 근거: Run104 showed the promoted mish stride24 max_steps100 candidate can achieve competitive validation on seed707, but with a high final_generalization_gap of 0.050492 and overfit_score of 0.216414. Run105 appeared to rescue the gap at stride20, but inspection of the consumed config showed the runner merged the partial LLM override onto an overfit-mitigation base, changing weight_decay, n_layers, drop_rate, and ffn_mult and reducing parameter_count to 215296. To isolate the stride hypothesis cleanly, this run explicitly fixes the architecture, regularization, activation, optimizer, data window size, and 100-step horizon while changing only seed707 and stride20 relative to the promoted baseline.
- 바꾼 변수: `{"max_steps": 100, "seed": 707, "stride": 20}`
- 기대 결과: A clean stride20 rescue should keep parameter_count near 413184, reduce seed707 final_generalization_gap below run104's 0.050492, lower overfit_score below 0.10, and ideally keep final_val_loss below 5.545 with fit_status=generalizing. If validation degrades toward run105's 5.548 band or overfit remains high, stride20 is not a sufficient matched-architecture rescue under the longer horizon.
- 실제 결과: final_val_loss=5.539270559946696, gap=0.012868563334147431, overfit_score=0.09433031082153409, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0129, overfit_score=0.0943. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If matched-architecture stride20 rescues seed707 while preserving validation, keep mish stride24 max_steps100 as the default for low-risk seeds and document stride20 as the first rescue for high-gap fresh seeds before testing one additional fresh seed. / 과적합 시 If matched-architecture stride20 still overfits seed707, test a shorter max_steps=95 horizon on seed707 with stride24 before changing activation, capacity, dropout, or weight decay.

## run 107 - 2026-06-03T04:04:25+00:00

- 보고서: `docs/train/runs/run_107.md`
- 이번 가설: For the seed707 mish stride24 case that overfit at max_steps=100, shortening the horizon to max_steps=95 will preserve most of the strong validation loss while reducing the late train-side overfit signal.
- 근거: Run104 showed that the promoted mish stride24 max_steps100 configuration had excellent raw validation on seed707 (final_val_loss=5.533458) but failed the risk gate with final_generalization_gap=0.050492 and overfit_score=0.216414. Run106 proved that stride20 can rescue the same seed under the matched 413184-parameter architecture, lowering the gap to 0.012869 and overfit_score to 0.094330, but it paid a validation-loss cost (final_val_loss=5.539271) and stayed medium risk. Since the 100-step horizon is strong on seeds151/202 but too aggressive on seed707, a smaller optimization-only change is the next safest test: keep stride24, mish, architecture, regularization, and data windowing fixed, but stop at 95 steps to see whether the raw validation advantage can be retained before the overfit score spikes.
- 바꾼 변수: `{"max_steps": 95, "seed": 707}`
- 기대 결과: If seed707's run104 failure is mainly late-horizon over-optimization, max_steps=95 should keep final_val_loss in the 5.533-5.539 band while reducing final_generalization_gap below 0.03 and overfit_score below 0.10. A successful result would beat run106 on validation while staying generalizing or low-medium risk. If overfit_score remains high, stride20 is the better rescue than shorter stride24 training.
- 실제 결과: final_val_loss=5.537662823994954, gap=0.043881495793660186, overfit_score=0.19658279418945224, fit_status=overfit_risk
- 과적합 판단: 과적합 위험. final gap=0.0439, overfit_score=0.1966. 다음 실험은 regularization 강화가 우선이다.
- 다음 가설: 성공 시 If max_steps=95 rescues seed707 with better validation than stride20, test max_steps=95 on seed151 or seed202 to see whether a slightly shorter horizon can become the safer default without losing run102's best score. / 과적합 시 If max_steps=95 still overfits seed707, keep max_steps100 as the low-risk-seed default and document stride20 as the first rescue path for high-gap fresh seeds before trying another fresh seed variance probe.

## run 108 - 2026-06-03T04:09:12+00:00

- 보고서: `docs/train/runs/run_108.md`
- 이번 가설: A fresh seed808 run with the promoted mish stride24 max_steps100 candidate will show whether seed707 is an outlier overfit case or whether the new 100-step default has a broader fresh-seed variance problem.
- 근거: Runs101-103 made max_steps100 the leading default candidate: seed606 improved raw validation, seed151 became the overfit-aware best, and seed202 stayed low-risk with the strongest raw validation. Seed707 then exposed a high-gap failure at stride24 in run104, and run107 showed that shortening stride24 to max_steps95 still overfits. The matched-architecture stride20 rescue in run106 reduced seed707's gap and overfit_score, but at a small validation cost. The next highest-information step is therefore seed variance, not another seed707 tweak: keep architecture, activation, stride, regularization, optimizer, and max_steps100 fixed, and test a new fresh seed to decide whether stride20 should remain a targeted rescue or become a more central policy.
- 바꾼 변수: `{"max_steps": 100, "seed": 808}`
- 기대 결과: If seed707 is mostly an outlier, seed808 should finish generalizing with final_val_loss in the 5.53-5.55 band, final_generalization_gap below 0.03, and overfit_score below 0.10. If seed808 also shows high gap or overfit_score above 0.15, the 100-step stride24 default has a broader fresh-seed overfit issue and stride20 rescue should be tested immediately on the same seed.
- 실제 결과: final_val_loss=5.536325136820476, gap=0.0038559834162397166, overfit_score=0.022364656130473115, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0039, overfit_score=0.0224. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If seed808 is low-risk, preserve mish stride24 max_steps100 as the default for low-risk seeds, document seed707 as a rescue case, and consider one more fresh seed only if confidence remains unclear. / 과적합 시 If seed808 overfits, keep seed808 fixed and test stride20 under the same 413184-parameter mish max_steps100 configuration before changing activation, capacity, learning rate, or regularization.

## run 109 - 2026-06-03T04:14:31+00:00

- 보고서: `docs/train/runs/run_109.md`
- 이번 가설: For the low-risk fresh seed808 mish stride24 candidate, increasing max_steps from 100 to 105 will test whether low-gap seeds remain mildly optimization-limited and can improve validation without entering the seed707-style overfit regime.
- 근거: Run108 showed that the promoted mish stride24 max_steps100 default generalizes on a fresh seed808 with final_val_loss=5.536325, final_generalization_gap=0.003856, train_val_improvement_gap=0.009254, and overfit_score=0.022365. This supports treating seed707 as a targeted rescue case, but run108 is still only slightly behind the overfit-aware best run102 (final_val_loss=5.534507). Because seed808's gap and overfit score are low, a tiny +5 step optimization probe is a safer and higher-information next test than changing activation, capacity, regularization, or stride. The run remains hardware-light on MPS and explicitly preserves the 413184-parameter Transformer configuration.
- 바꾼 변수: `{"max_steps": 105, "seed": 808}`
- 기대 결과: If seed808 is still mildly optimization-limited, max_steps105 should lower final_val_loss below run108's 5.536325, ideally near or below run102's 5.534507, while keeping final_generalization_gap below 0.02 and overfit_score below 0.08. If train loss improves but validation stalls or overfit_score rises above 0.10, max_steps100 should remain the low-risk default horizon.
- 실제 결과: final_val_loss=5.533207734425862, gap=0.012854417165120147, overfit_score=0.049359957377114405, fit_status=generalizing
- 과적합 판단: 일반화 개선 신호. final gap=0.0129, overfit_score=0.0494. seed 반복으로 재현성을 확인할 만하다.
- 다음 가설: 성공 시 If max_steps105 improves seed808 without medium-risk overfit, test the same 105-step horizon on seed151 or seed202 before considering it as a low-risk-seed refinement. / 과적합 시 If max_steps105 overfits or worsens validation, keep mish stride24 max_steps100 as the default and reserve stride20 only for high-gap rescue cases such as seed707.
