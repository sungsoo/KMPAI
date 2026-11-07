# 실험 설계 변경 이력

결과를 본 뒤의 변경은 논문 한계 절에도 쓴다.

## 2026-11-06T21:36:03
- 사유: pre-data clarification found in the Linux pipeline check (6 smoke cases, no main data): M1 and M2 now state when a claim counts as supported (bound-based), so absence of evidence is reported as inconclusive rather than supported; thresholds and data unchanged
- 이전 해시: 1527aee13baf → 새 해시: c209be719897
- 변경: claims[M1].falsify_if: "on a system the one-sided 95% Clopper-Pearson upper bound of the worse share is below 0.10: not supported for that system" → "per system: supported if the one-sided 95% Clopper-Pearson LOWER bound of the worse share is at least 0.10; not supported if the one-sided 95% UPPER bound is below 0.10; inconclusive otherwise"
- 변경: claims[M2].falsify_if: "on a system the one-sided 95% Clopper-Pearson lower bound of the share exceeds 0.10: not supported; fewer than 10 accepted runs: inconclusive" → "per system: fewer than 10 accepted runs: inconclusive; not supported if the one-sided 95% Clopper-Pearson LOWER bound exceeds 0.10; supported if the one-sided 95% UPPER bound is at most 0.10; inconclusive otherwise"
