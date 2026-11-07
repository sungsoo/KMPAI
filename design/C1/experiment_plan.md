# 실험 설계서: C1: constraint-preserving correction and acceptance testing for multi-joint arm world models

- 대상: ICT Express 2027 original research (manuscript-v1 line)
- 잠금: c209be719897 (2026-11-06T21:36:03)

## 연구 질문

- RQ1: Does a constraint-preserving (contraction) correction still harm forecast accuracy on multi-joint arms (d = 2, 3, 7)?
- RQ2: Does LTT acceptance keep accepted runs harmless in-distribution and under range widening as the number of joints grows?
- RQ3: Does cross-joint coupling make the full correction matrix the useful candidate as d grows, and what does it add over per-joint (diagonal) weights?
- RQ4: How does the judge-set size needed for acceptance change with d (n = 36, 60, 100)?

## 데이터

| 이름 | 분할 | 분할 단위 | 출처 | 라이선스 |
|---|---|---|---|---|
| planar_2link | per run: train 12/36, C_f 12, C_s1 12, C_s2 36 (60/100 post hoc), test 48 episodes; test generated with separate seeds | group | self-implemented planar 2-link arm (Lagrangian rigid-body dynamics), nominal parameters as physics prior; truth adds joint friction (Coulomb + viscous), link-mass error and joint coupling | own code |
| planar_3link | as planar_2link | group | self-implemented planar 3-link arm, same construction as planar_2link | own code |
| franka_7dof | as planar_2link | group | MuJoCo 4.0.0 (pip) with franka_emika_panda/panda_nohand.xml from MuJoCo Menagerie, commit 98a70c569c4c1b6da467a20012f766ee1f8dc521 (pinned 2026-11-06); prior = nominal model without joint friction/damping mismatch; truth = model with friction, armature and link-mass error | MuJoCo Apache-2.0; franka_emika_panda LICENSE Apache-2.0 (checked at the pinned commit, 2026-11-06) |

## 비교군

| 방법 | 역할 | 출처 / 제거 요소 | 선정 이유 |
|---|---|---|---|
| ltt_correction | proposed | this work (Algorithm 1 of manuscript-v1, unchanged), candidates {0, alpha I, D_lambda, A_lambda} |  |
| equal_budget_retrain | baseline | residual MLP retrained on all non-test episodes with the same settings | the alternative a deployer with weight access would use; strongest accuracy competitor in the 2-DOF pilot |
| mean_loss_rule | baseline | apply the C_s1-best candidate if it lowers the C_s1 loss by 1% | simplest validation rule; matched LTT's mean gain in the 2-DOF pilot, so it is the fair test of whether the finite-sample gate adds anything |
| per_step_combination | baseline | per-step combination weights fitted on C_f (Wang et al., Int. J. Forecast. 2023) | forecast-combination baseline that breaks kinematic constraints |
| keep_Z | simple_baseline | learned world model forecast without correction | the incumbent every correction must not worsen |
| physics_prior | simple_baseline | nominal-model rollout | lower anchor |
| no_contraction | ablation | contraction |  |
| fixed_matrix_no_gate | ablation | ltt_gate |  |
| ltt_diagonal_only | ablation | full_matrix_candidate |  |

## 지표

- median_rel_gain (높을수록 좋음) **주 지표**
- accepted_worse_share_id (낮을수록 좋음) **주 지표**
- worse_share_range_widening (낮을수록 좋음)
- full_matrix_pick_share (높을수록 좋음)

## 주장과 판정 기준

| ID | 주장 | 비교 | 최소 의미 효과 | 반증 기준 | 기대(결과 아님) |
|---|---|---|---|---|---|
| M1 | On every multi-joint system, the always-applied contraction correction is worse than keep_Z in at least 10% of runs under range widening | fixed_matrix_no_gate vs keep_Z | 0.1 (one run in ten made worse is operationally unacceptable for a deployed forecaster; the 2-DOF pilot showed 0.49) | per system: supported if the one-sided 95% Clopper-Pearson LOWER bound of the worse share is at least 0.10; not supported if the one-sided 95% UPPER bound is below 0.10; inconclusive otherwise | 기대(결과 아님): 일부 시스템에서 지지 |
| M2 | Among runs that LTT accepts, the share worse than keep_Z in-distribution stays at or below delta = 0.1 on every system | ltt_correction vs keep_Z | 0.1 (delta of Proposition 2) | per system: fewer than 10 accepted runs: inconclusive; not supported if the one-sided 95% Clopper-Pearson LOWER bound exceeds 0.10; supported if the one-sided 95% UPPER bound is at most 0.10; inconclusive otherwise | 기대(결과 아님): 지지 |
| M3 | On franka_7dof, LTT gives a median per-run relative gain over keep_Z of at least 5% across all runs (rejected runs count as 0) | ltt_correction vs keep_Z | 0.05 (below 5% a deployer would not pay for 60 observed episodes per acceptance) | the lower end of the seed-cluster bootstrap 95% interval is below 0.05: not supported | 기대(결과 아님): 불확실 |
| M4 | On franka_7dof, adding the full-matrix candidate raises the median gain over the diagonal-only gate by at least 2 points | ltt_correction vs ltt_diagonal_only | 0.05 (cross-joint coupling is the reason to fit a d x d matrix (49 parameters at d = 7 against 7 for diagonal); a benefit below 5 points does not justify them, and a 2-point effect would need about 186 seeds (plan_tool check)) | Holm-adjusted Wilcoxon p >= 0.05 or the median paired difference is below 0.05: not supported | 기대(결과 아님): 불확실 |
| M5 | Descriptive: the share of accepted corrections that are full matrices grows with d (2, 3, 7), and the acceptance rate as a function of judge-set size n = 36, 60, 100 | ltt_correction vs ltt_diagonal_only | 0.2 (a 20-point change between d = 2 and d = 7 would change design advice) | descriptive; reported as observed without a verdict | 기대(결과 아님): 증가 |

## 통계와 반복

- 반복 단위 seed, 값 [3011, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313] (사유: 34 seeds: power 0.8 at alpha 0.05 for M1 (sd 0.162, min effect 0.10) and M3 (sd 0.079, min effect 0.05), sds from the 2-DOF pilot (results_h10s_confirm). Seed values mod 1000 = 11..313 differ from every earlier protocol (101..229, 301..423, 503..661, 511..621, 701..857); the planar arms and the Franka model also use different data generators from the 2-DOF oscillators)
- 구간 bootstrap95, 대응 True, 검정 wilcoxon, alpha 0.05, 다중 비교 보정 holm
- 판정 supported = 차이 구간이 주장 방향으로 0을 벗어나고 평균 차이 ≥ min_effect

| 주장 | 예상 sd | min_effect | 검정력 80% 필요 seed | 현재 |
|---|---|---|---|---|
| M1 | 0.162 | 0.1 | 34 | 34 |
| M2 | 0 | 0.1 | 추정 불가 | 34 |
| M3 | 0.079 | 0.05 | 33 | 34 |
| M4 | 0.079 | 0.05 | 33 | 34 |
| M5 | 0 | 0.2 | 추정 불가 | 34 |

## 예산

- 실행 918개 x 0.2분 = 184분, 튜닝 5분, 허용 300분, 장비 Apple M5 Max 128GB (primary, parallel cases) via Claude Code; Linux 1 CPU core for a reduced smoke subset and a cross-platform decision check
- 튜닝: 방법당 1회, 분할 validation

## 중단 규칙

- 예산의 80% 초과 시 seed를 줄이지 않고 franka_7dof의 학습 에피소드 설정(12/36) 중 하나를 줄인 뒤 기록
- 버그 발견 시 영향받는 실행만 재실행하고 CHANGELOG와 plan_changes.md에 기록
- 실패/시간 초과 실행은 0이 아니라 결측으로 기록하고 분모에서 제외한 수를 보고
- 결과를 본 뒤의 변경은 relock 사유와 함께 논문 한계 절에 기록

## 검사 결과

- 경고: M2: 지표 expected_sd 없음 → 필요한 seed 수 추정 불가(파일럿 실행 권장)
- 경고: M5: 지표 expected_sd 없음 → 필요한 seed 수 추정 불가(파일럿 실행 권장)
- 경고: 합성 데이터: 논문 본문/캡션/한계에 synthetic 명시, 실제 환경 주장 금지
