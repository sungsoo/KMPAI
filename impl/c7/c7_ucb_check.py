"""Post hoc check (round 13 review, P2): does the betting bound's definition matter?
c7_rules.betting_ucb returns the smallest grid point m whose null H0: E x >= m is rejected. The textbook confidence
bound is the largest grid point that is NOT rejected. The two coincide when rejection is monotone in m. This script
recomputes every Study 2 decision (whole candidate and every direction prefix, n = 100 and n = 36) with both
definitions and reports disagreements; it changes no locked code or result."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; IMPL = HERE.parent
sys.path.insert(0, str(HERE))
from c7_rules import betting_ucb


def ucb_conservative(x, eps=0.05, grid=2001):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0); n = len(x)
    mu_hat = np.concatenate([[0.5], np.cumsum(x)[:-1] / np.arange(1, n)])
    var_hat = np.concatenate([[0.25], np.array([np.var(x[:t]) if t > 1 else 0.25 for t in range(1, n)])]) + 1e-4
    lam_base = np.sqrt(2 * np.log(1 / eps) / (n * var_hat)); keep = 0.0
    for m in np.linspace(0.0, 1.0, grid)[1:]:
        lam = np.minimum(lam_base, 0.5 / max(1 - m, 1e-12))
        if np.prod(1 + lam * (m - x)) < 1 / eps:
            keep = m
    return float(min(1.0, keep + 1.0 / (grid - 1)))


def main(suffix='_mac'):
    reps = [json.loads(f.read_text()) for f in sorted((IMPL / f'results_c7{suffix}' / 'cases').glob('*/complete.json'))]
    n_dec = dis = 0; worst = 0.0
    for r in reps:
        for ratios in [r['judge_ratio_whole']] + r['judge_ratio_prefixes']:
            for n in (100, 36):
                x = np.minimum(np.array(ratios[:n]), 2.0) / 2.0
                a, b = betting_ucb(x), ucb_conservative(x)
                n_dec += 1; dis += (a <= 0.5) != (b <= 0.5); worst = max(worst, abs(a - b))
    t0 = time.perf_counter(); x = np.minimum(np.array(reps[0]['judge_ratio_whole']), 2) / 2
    for _ in range(50):
        betting_ucb(x)
    ms = 1000 * (time.perf_counter() - t0) / 50
    out = {'decisions_checked': n_dec, 'decision_disagreements': int(dis), 'max_bound_difference': worst, 'ms_per_bound_n100_linux_1core': ms}
    (IMPL / f'results_c7{suffix}' / 'analysis' / 'ucb_check.json').write_text(json.dumps(out, indent=2)); print(out)


if __name__ == '__main__':
    main()
