"""C2 (reduced): ICT cost of the acceptance procedure on this machine (descriptive measurement; no locked
decision or result changes).

    python c7/c7_ict_cost.py [--suffix _mac] [--out PATH] [--repeats 200]

Measures, per arm (d = 2, 3, 7):
  1. gate time: the magnitude rule with directions on stored Study 2 judge ratios (100 episodes; whole candidate
     and every direction prefix, as in impl/c7/c7_study.py), median and 95th percentile over cases and repeats;
  2. ratio time: computing the 100 episode losses of one corrected forecast (P + R A and its MSE) on arrays of the
     study's shape (100 episodes x 100 steps x 2d), median over repeats;
  3. apply time: correcting one forecast (100 steps x 2d) with a d x d matrix;
  4. data volume (bytes, float32): observed judge trajectories Y (100 x 101 x 2d), forecasts P and Z of the judge
     episodes, and the matrix A that a deployment ships.
Writes JSON with the machine description so Linux and Mac numbers can be reported side by side.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; IMPL = HERE.parent
for p in (IMPL, IMPL / 'c1', HERE):
    sys.path.insert(0, str(p))
from c7_rules import magnitude_gate
from c1_models import apply

ARMS = {'planar_2link': 2, 'planar_3link': 3, 'franka_7dof': 7}


def gate_decision(whole, prefixes, n=100):
    """Magnitude rule with directions, exactly as in c7_study (stop at the first rejected prefix)."""
    accepted = 0
    for r in prefixes if prefixes else [whole]:
        if not magnitude_gate(np.array(r[:n]))[0]:
            break
        accepted += 1
    return accepted


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suffix', default='_mac'); ap.add_argument('--out', default=None)
    ap.add_argument('--repeats', type=int, default=200); ap.add_argument('--cases', type=int, default=20)
    a = ap.parse_args()
    reps = [json.loads(f.read_text()) for f in sorted((IMPL / f'results_c7{a.suffix}' / 'cases').glob('*/complete.json'))]
    rng = np.random.default_rng(20261107); out = {'machine': {'platform': platform.platform(), 'machine': platform.machine(),
           'processor': platform.processor(), 'python': platform.python_version(), 'numpy': np.__version__}, 'arms': {}}
    for arm, d in ARMS.items():
        cases = [r for r in reps if r['system'] == arm][:a.cases]
        gt = []
        for r in cases:
            for _ in range(max(1, a.repeats // 20)):
                t0 = time.perf_counter(); gate_decision(r['judge_ratio_whole'], r['judge_ratio_prefixes']); gt.append(1e3 * (time.perf_counter() - t0))
        P = rng.normal(size=(100, 100, 2 * d)).astype(np.float64); Z = P + 0.01 * rng.normal(size=P.shape); Y = P + 0.02 * rng.normal(size=P.shape)
        A = np.eye(d) * 0.5; rt = []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); c = apply(P, Z, A, d); ((c - Y) ** 2).mean(axis=(1, 2)); rt.append(1e3 * (time.perf_counter() - t0))
        one_p, one_z = P[:1], Z[:1]; at = []
        for _ in range(a.repeats * 5):
            t0 = time.perf_counter(); apply(one_p, one_z, A, d); at.append(1e6 * (time.perf_counter() - t0))
        b = 4
        out['arms'][arm] = {'d': d, 'cases_timed': len(cases),
                            'gate_ms_median': float(np.median(gt)), 'gate_ms_p95': float(np.percentile(gt, 95)),
                            'ratios_ms_median_per_candidate': float(np.median(rt)), 'apply_us_median_per_forecast': float(np.median(at)),
                            'bytes_observed_judge_Y': 100 * 101 * 2 * d * b, 'bytes_judge_forecasts_PZ': 2 * 100 * 100 * 2 * d * b,
                            'bytes_matrix_A': d * d * b}
    path = Path(a.out) if a.out else IMPL / f'results_c7{a.suffix}' / 'analysis' / f"ict_cost_{platform.system().lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(out, indent=2))
    for arm, v in out['arms'].items():
        print(f"{arm}: gate {v['gate_ms_median']:.1f} ms (p95 {v['gate_ms_p95']:.1f}), ratios {v['ratios_ms_median_per_candidate']:.2f} ms, apply {v['apply_us_median_per_forecast']:.1f} us, "
              f"judge Y {v['bytes_observed_judge_Y'] / 1e3:.0f} kB, A {v['bytes_matrix_A']} B")
    print('->', path)


if __name__ == '__main__':
    main()
