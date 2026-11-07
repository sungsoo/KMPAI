# 실험 설계서: C7: magnitude-based and direction-wise acceptance of constraint-preserving corrections for multi-joint arm world models

- 대상: ICT Express 2027 original research (manuscript-v1 line)
- 잠금: 39289bba1c10 (2026-11-06T22:32:36)

## 연구 질문

- RQ1: Does accepting a correction by the size of its loss change (an upper confidence bound on the mean relative loss) recover safe gains that the strict count rule of C1 gave up on a 7-DOF arm?
- RQ2: Do accepted corrections stay harmless in-distribution under the magnitude rule on every system?
- RQ3: Does direction-wise partial acceptance (fixed-sequence testing over eigen-directions of the contraction) accept more useful corrections than whole-matrix acceptance?
- RQ4: What do the judge-set size (36 vs 100) and a tail condition cost or buy, and what happens under range widening (not covered by the guarantee)?

## 데이터

| 이름 | 분할 | 분할 단위 | 출처 | 라이선스 |
|---|---|---|---|---|
| planar_2link | per run: train 12/36, C_f 12, C_s1 12, judge 100 (first 36 = small-judge variant), test 48 ID + 48 range widening | group | same generator, prior, truth and settings as C1 (impl/c1/c1_models.py, c1_config.json) with the C7 seeds | own code |
| planar_3link | as planar_2link | group | as planar_2link | own code |
| franka_7dof | as planar_2link | group | as C1 (MuJoCo 4.0.0, Menagerie commit 98a70c56, panda_nohand.xml) | MuJoCo Apache-2.0; franka_emika_panda Apache-2.0 |

## 비교군

| 방법 | 역할 | 출처 / 제거 요소 | 선정 이유 |
|---|---|---|---|
| mag_dir_gate | proposed | this work: rule (1) magnitude gate (one-sided 95% betting upper confidence bound, Waudby-Smith and Ramdas 2024, on the mean of min(ratio, 2)/2 over the n = 100 judge episodes (primary); accept iff bound <= (1 + tau)/2, tau = 0; no tail condition in the primary rule) applied by rule (2) fixed-sequence testing over the eigen-directions of A - I ordered by their C_s1 loss reduction; the accepted prefix gives A_S = I + sum_{j in S} (lambda_j - 1) v_j v_j^T, which stays symmetric with eigenvalues in [0, 1] (Proposition 1 keeps the linear constraints) |  |
| strict_ltt_c1 | baseline | C1 rule: whole C_s1-best candidate, accept iff Clopper-Pearson upper bound on P(episode worse) <= 0.1 with 36 judge episodes | the rule whose conservativeness C7 addresses; same cases |
| mean_loss_rule | baseline | apply the C_s1-best candidate if it lowers the C_s1 loss by 1% (no finite-sample test) | simplest validation rule; shows what the guarantee costs |
| equal_budget_retrain | baseline | as C1 | most accurate alternative when weights are available |
| keep_Z | simple_baseline | learned world model forecast | incumbent |
| always_on | simple_baseline | C_s1-best candidate always applied | upper reference for the gain a gate can recover |
| mag_whole_gate | ablation | direction_wise |  |
| strict_dir_gate | ablation | magnitude_rule |  |

## 지표

- median_rel_gain (높을수록 좋음) **주 지표**
- accepted_mean_ratio_worse_share_id (낮을수록 좋음) **주 지표**
- runs_with_accepted_correction (높을수록 좋음)

## 주장과 판정 기준

| ID | 주장 | 비교 | 최소 의미 효과 | 반증 기준 | 기대(결과 아님) |
|---|---|---|---|---|---|
| N1 | On franka_7dof in-distribution, mag_dir_gate gives a median per-run relative gain over keep_Z of at least 5% (rejected runs count as 0) | mag_dir_gate vs keep_Z | 0.05 (same threshold as C1-M3, so the two rules are judged by the same bar; below 5% a deployer would not pay for the observed judge episodes) | lower end of the seed-cluster bootstrap 95% interval below 0.05: not supported | 기대(결과 아님): 불확실(파일럿 12 케이스에서 Franka 승인 일부) |
| N2 | Per system, among runs where mag_dir_gate applies a correction, the share whose in-distribution test mean ratio (mean over test episodes of min(ratio, 2)) exceeds 1 is at most 0.10 | mag_dir_gate vs keep_Z | 0.1 (epsilon-level guarantee of rule (1) translated to runs; 0.10 matches delta of C1) | per system: fewer than 10 runs with an applied correction: inconclusive; not supported if the one-sided 95% Clopper-Pearson LOWER bound exceeds 0.10; supported if the one-sided 95% UPPER bound is at most 0.10; inconclusive otherwise | 기대(결과 아님): 불확실 |
| N3 | On franka_7dof, direction-wise acceptance applies a correction in more runs than whole-matrix acceptance under the same magnitude rule, by at least 10% of runs | mag_dir_gate vs mag_whole_gate | 0.1 (14 of 136 runs; fewer would not justify the extra procedure) | difference below 10% of runs or Holm-adjusted one-sided exact McNemar p >= 0.05: not supported | 기대(결과 아님): 불확실 |
| N4 | Descriptive: acceptance rates and median gains of all gates per system; the 36-episode judge variant and the tail-condition variant (P(ratio > 1.1) Clopper-Pearson upper bound <= 0.2); comparison with C1's strict rule on the same cases | mag_dir_gate vs strict_ltt_c1 | 0.1 (descriptive; a 10-point change in acceptance rate between gates or judge sizes would change design advice) | descriptive; reported as observed without a verdict | 기대(결과 아님): 엄격 규칙보다 승인 증가 |
| N5 | Descriptive: under range widening (not covered by the guarantee), the share of runs with an applied correction that become worse than keep_Z, per system and gate | mag_dir_gate vs keep_Z | 0.1 (descriptive; one accepted run in ten made worse under range widening would be operationally relevant) | descriptive; reported as observed without a verdict | 기대(결과 아님): 평면 팔에서 악화 발생 가능 |

## 통계와 반복

- 반복 단위 seed, 값 [6901, 6904, 6907, 6910, 6913, 6916, 6919, 6922, 6925, 6928, 6931, 6934, 6937, 6940, 6943, 6946, 6949, 6952, 6955, 6958, 6961, 6964, 6967, 6970, 6973, 6976, 6979, 6982, 6985, 6988, 6991, 6994, 6997, 6999] (사유: 34 new seeds (mod 1000 = 901..999, unused by every earlier protocol including C1 = 11..313). Power: seed-level sd of the 7-DOF ID gain of the always-on contraction in C1 was 0.058 (used as a pilot value; the gated gain has more zeros, so 0.079 from the 2-DOF pilot is used as a conservative sd))
- 구간 bootstrap95, 대응 True, 검정 wilcoxon, alpha 0.05, 다중 비교 보정 holm
- 판정 supported = 차이 구간이 주장 방향으로 0을 벗어나고 평균 차이 ≥ min_effect

| 주장 | 예상 sd | min_effect | 검정력 80% 필요 seed | 현재 |
|---|---|---|---|---|
| N1 | 0.079 | 0.05 | 33 | 34 |
| N2 | 0 | 0.1 | 추정 불가 | 34 |
| N3 | 0 | 0.1 | 추정 불가 | 34 |
| N4 | 0.079 | 0.1 | 11 | 34 |
| N5 | 0 | 0.1 | 추정 불가 | 34 |

## 예산

- 실행 816개 x 0.2분 = 163분, 튜닝 5분, 허용 300분, 장비 Apple M5 Max 128GB (primary, parallel cases) via Claude Code; Linux 1 CPU core for a reduced smoke subset and a cross-platform decision check
- 튜닝: 방법당 1회, 분할 validation

## 중단 규칙

- 예산의 80% 초과 시 seed를 줄이지 않고 학습 크기 설정 하나를 줄인 뒤 기록
- 버그 발견 시 영향받는 실행만 재실행하고 CHANGELOG와 plan_changes.md에 기록
- 실패 / 시간 초과 실행은 결측으로 기록하고 분모에서 제외한 수를 보고
- 결과를 본 뒤의 변경은 relock 사유와 함께 논문 한계 절에 기록

## 검사 결과

- 경고: N2: 지표 expected_sd 없음 → 필요한 seed 수 추정 불가(파일럿 실행 권장)
- 경고: N3: 지표 expected_sd 없음 → 필요한 seed 수 추정 불가(파일럿 실행 권장)
- 경고: N5: 지표 expected_sd 없음 → 필요한 seed 수 추정 불가(파일럿 실행 권장)
- 경고: 합성 데이터: 논문 본문/캡션/한계에 synthetic 명시, 실제 환경 주장 금지
